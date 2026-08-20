"""Fail when a case or data-source token leaks into a generic position.

A token naming one specific case or forcing source (a campaign date, a city,
an upstream model acronym) is legitimate in case data, in a source-specific
adapter, and in test fixtures.  It is never legitimate in a *generic*
position, because a model quietly specialized to one data source cannot be
re-pointed at another without a rewrite, and the specialization stays
invisible until someone tries.

This gate therefore does not scan everything.  It scans the zones where the
rule says a case token must never appear:

  * the numerical core,
  * configuration defaults and schema,
  * physics identifiers and the physics registry,
  * generically-selected runner names.

Within those zones only *code* positions are checked -- identifiers and
identifier-shaped string literals.  Comments and docstrings are ignored on
purpose: a provenance note recording which run pinned a constant is
documentation, not specialization, and rewriting those in bulk would bury the
signal this gate exists to surface.

Suppression is position-aware
-----------------------------
Every occurrence carries a *position* as well as a file and a spelling, and
the baseline is keyed on all three::

    <relative path>::<position>::<kind>::<text>

where the position is

  * a RFC 6901 JSON Pointer for ``.json`` documents,
  * ``<scope qualname>#<ordinal>`` for Python, where the ordinal counts
    occurrences of that same (kind, text) inside that same scope in source
    order, and
  * ``#<ordinal>`` for Rust, counted over the whole file.

Positions are deliberately line-independent: a baseline keyed on line numbers
would fire on every unrelated edit above it, and a gate that cries wolf gets
switched off.  Counting occurrences instead makes the key stable under moves
and still sensitive to *addition* -- a second ``hrrr`` in a scope that had one
is ordinal ``#2``, which no baseline entry covers, so it is reported.

An earlier version of this gate keyed suppression on ``<path>::<text>`` alone.
That is position-blind: one baseline entry for ``hrrr`` in the physics
registry suppressed ``hrrr`` *anywhere* in that document, including as a
physics-template id or a component option key -- the two positions this
module's own rules call illegitimate.  It happened: commit ``af04478`` added a
source id to a route and the gate printed a clean bill of health.  Do not
reintroduce a key that drops the position.

There is also no shape-based auto-suppression any more.  A rule of the form
"anything under ``source_ids`` is data, therefore fine" re-opens exactly that
hole one level up: the next source id appended to that list is absorbed
without anyone reading it.  Declaring that a route serves a given source
really is legitimate, but it is legitimate *one entry at a time*, so each such
entry earns a baseline line with a written reason instead.

Imports are checked, including through the name
---------------------------------------------
A per-file scanner that only reads identifiers and literals cannot see a
specialization that arrives by import.  Two shapes get through:

  * ``import gpuwm.ingest.hrrr`` / ``from gpuwm.ingest.hrrr import Snapshot``
    in a protected module -- the token is in the ``Import`` node, which is
    neither a ``Name`` nor a string constant, so nothing recorded it; and

  * ``from gpuwm.verify.nest_gates import CHILD_REFERENCE_FRAMES``, where the
    *name* is perfectly generic and the *value* it is bound to is one case's
    frame table.  The importing file reads clean to any scanner that never
    opens the file it imports from.

Both are now recorded, the second by resolving the import to a file inside
the scanned root, finding the module-scope binding of the imported name, and
scanning the expression it is bound to (following re-exports, with a visited
set).  Only in-tree modules are resolved -- third-party packages are not this
gate's business -- and a name bound to a function or a class is not followed,
because the rule is about specialized *data* reaching generic code.

Usage
-----
    python tools/check_case_token_leakage.py <path> [<path> ...]
    python tools/check_case_token_leakage.py <path> --emit-baseline

An argument is a path to scan, NOT a root the zones are spelled against.
Every zone above is relative to a REPOSITORY root, so each argument is
resolved by walking up to the nearest directory that actually holds one of
them; the argument then narrows the scan to the zones inside it.  ``.`` and
``gpuwm`` and ``gpuwm/core/dyn`` are all lawful, and all report paths -- and
therefore baseline keys -- relative to the repository.

THE BREAKAGE THAT NAMES THIS, measured at ``b858ad824``: the arguments used
to be handed to :func:`scan_root` as roots.  With ``root=gpuwm`` the zone
``gpuwm/core/`` resolved to ``gpuwm/gpuwm/core``, which does not exist, so
``python tools/check_case_token_leakage.py gpuwm tools tests docs`` -- the
line every lane and every close/verify wave ran by hand -- scanned ZERO
files and printed the same clean bill of health a clean tree prints.  Months
of hand-runs and reports quoted a scan of nothing as a green gate.  So the
count of files scanned is now printed on every run, and a target that
selects no zone is a refusal rather than a pass.

Exits non-zero when an occurrence appears that the baseline does not name.
Known pre-existing occurrences live in the sidecar baseline, each with a
one-line reason.  A reason that is empty, or that still starts with ``TODO``
or ``UNREVIEWED``, does not suppress anything -- baselining is a decision
somebody has to write down.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Sequence

# Tokens naming one case or one forcing source.  Matched case-insensitively.
# Purely numeric tokens are only a violation when glued into a longer
# identifier (``case1974``); a bare year inside a date literal is not.
CASE_TOKENS: tuple[str, ...] = (
    "real74",
    "hrrr",
    "ohio",
    "oklahoma",
    "may1999",
    "20cr",
    "1974",
    # Observation-battery case days.  A battery case IS its UTC day, so the
    # day is the whole of its identity and there is nothing else to guard:
    # it lives in configs/battery/case_<day>.toml, in the receipts that
    # measure that day, and here.  Registered with the case, per the battery
    # spec section 1.1 ("every new battery case token is appended to
    # CASE_TOKENS in the same commit that introduces the case").
    "20240521",
    # The tornado-scale LES attempt's named case (G1, ratified 2026-08-05).
    # Registered on the same footing as a battery case day and for the same
    # reason: the case IS its place and its UTC day, so those two tokens are
    # the whole of its identity. It is allowed exactly two homes --
    # configs/les_tornado_100m_dodgecity_20160524.toml and
    # gpuwm/verify/cases/les_tornado_dodgecity_20160524.py, plus the docs
    # that rule on it and the receipts that measure it -- and registering
    # the tokens here is what turns that sentence into a gate.
    "dodgecity",
    "dodge city",
    "20160524",
    # The case attempt #1 actually runs, after Drew moved it on 2026-08-06.
    # The Dodge City tokens above stay registered: that config and its case
    # module remain in the tree, parked behind an unadjudicated ingest
    # question, and a parked case needs its confinement enforced exactly as
    # much as a live one does.
    "mayfield",
    "20211210",
)

# Zones where a case token must never appear, relative to a repository root.
# A zone is a directory prefix or an exact file path.
PROTECTED_ZONES: tuple[str, ...] = (
    # numerical core and physics
    "gpuwm/core/",
    "gpuwm/physics_registry.py",
    "gpuwm/physics_registry_v2.json",
    "gpuwm/physics_compat.py",
    # configuration defaults and schema
    "gpuwm/config.py",
    "gpuwm/experiment.py",
    "gpuwm/runtime.py",
    "gpuwm/supervisor.py",
    # certification measurement path: the one certified ULP definition, the
    # generic t=0 full-state digest and its CLI.  A case token here would
    # mean the digest scores one campaign's staging layout rather than a
    # staged pair, which is the specialization this gate exists to catch.
    "gpuwm/verify/state_equiv.py",
    "gpuwm/verify/t0_state_digest.py",
    "tools/matched_wrfout_t0_state_digest.py",
    # the published certification receipts.  A receipt's *input inventory*
    # names the files that were staged, so a campaign spelling there is the
    # provenance the receipt exists to carry and earns a baseline line; a
    # campaign spelling anywhere else in a receipt -- a group name, a
    # variable, a registration field -- would mean the certified measurement
    # itself was written for one case, which is what this zone watches for.
    "gpuwm/data/certification/",
    # generic verification instruments: these score whatever they are
    # pointed at, so a case token here would silently specialize the
    # yardstick rather than the run being measured
    "gpuwm/verify/field_metrics.py",
    "gpuwm/verify/chaos_envelope.py",
    "gpuwm/verify/spectral.py",
    # the observation scorer: it consumes observations and model fields
    # through one seam and takes its case as data, so a case token anywhere
    # in it would mean the referee had been written for one day's weather.
    # A data-source token is equally disqualifying here -- the ingest side is
    # where a product name belongs, and by the time bytes reach the scorer
    # they are "a gridded field with a validity mask".
    "gpuwm/verify/obs/",
    "tools/matched_wrfout_envelope.py",
    "tools/certify_band_from_ensemble.py",
    # generic simulation layer and runner selection
    "crates/rw-sim/src/domain.rs",
    "crates/rw-sim/src/physics.rs",
    "crates/rw-sim/src/physics_v2.rs",
    "crates/rw-sim/src/headless.rs",
    "crates/rw-sim/src/runtime.rs",
    "crates/rw-sim/src/namelist.rs",
)

# Reason prefixes that do not count as a decision.  An entry carrying one of
# these is reported as if it were not in the baseline at all.
_UNDECIDED_REASON_PREFIXES: tuple[str, ...] = ("TODO", "UNREVIEWED", "FIXME", "XXX")

# Violations that already existed when this gate was introduced, kept in a
# sidecar file so the gate blocks regressions today while these are retired
# one at a time.  Never add to it to make a new violation pass.
BASELINE_PATH = Path(__file__).with_name("case_token_baseline.json")

MODULE_SCOPE = "<module>"


def reason_is_a_decision(reason: object) -> bool:
    """A baseline reason suppresses only when somebody actually wrote one."""
    if not isinstance(reason, str):
        return False
    stripped = reason.strip()
    if not stripped:
        return False
    upper = stripped.upper()
    return not any(upper.startswith(p) for p in _UNDECIDED_REASON_PREFIXES)


def reason_is_debt(reason: str) -> bool:
    """A carried entry that is a real violation, not a decision that it is data.

    Suppressing it keeps the gate usable for regressions; counting it keeps the
    debt visible on every run so it cannot quietly become permanent.
    """
    return reason.strip().upper().startswith("DEBT")


def load_baseline(path: Path = BASELINE_PATH) -> tuple[dict[str, str], str | None]:
    """Return ``(key -> reason, error)``.

    On any problem the baseline is empty, so the gate fails closed: every
    occurrence is reported rather than silently suppressed by a file nobody
    can read.
    """
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}, f"{path} is missing"
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return {}, f"{path} is unreadable: {exc}"

    known = document.get("known_violations")
    if not isinstance(known, dict):
        return {}, (
            f"{path}: 'known_violations' must be an object mapping each "
            "position-aware key to a one-line reason (the old list form is "
            "position-blind and is no longer accepted)"
        )
    return {
        str(key): str(reason)
        for key, reason in known.items()
        if reason_is_a_decision(reason)
    }, None


BASELINE, BASELINE_ERROR = load_baseline()

_ALPHANUMERIC_TOKENS = tuple(t for t in CASE_TOKENS if not t.isdigit())
_NUMERIC_TOKENS = tuple(t for t in CASE_TOKENS if t.isdigit())

_ALPHA_RE = (
    re.compile("|".join(re.escape(t) for t in _ALPHANUMERIC_TOKENS), re.IGNORECASE)
    if _ALPHANUMERIC_TOKENS
    else None
)
_NUMERIC_RE = (
    re.compile("|".join(re.escape(t) for t in _NUMERIC_TOKENS))
    if _NUMERIC_TOKENS
    else None
)

_IDENT_CHUNK_RE = re.compile(r"[A-Za-z0-9_]+")
_RUST_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_RUST_STRING_RE = re.compile(r'"(?:[^"\\]|\\.)*"')


def offending_token(name: str) -> str | None:
    """Return the case token that makes ``name`` a violation, if any."""
    if _ALPHA_RE is not None:
        match = _ALPHA_RE.search(name)
        if match is not None:
            return match.group(0)
    if _NUMERIC_RE is not None:
        # A numeric token only offends when embedded in a longer identifier,
        # so a date literal such as 1974-04-03 stays clean.
        for chunk in _IDENT_CHUNK_RE.findall(name):
            match = _NUMERIC_RE.search(chunk)
            if match is not None and chunk != match.group(0):
                return chunk
    return None


def literal_offence(text: str) -> str | None:
    """Case tokens in prose are documentation; only identifier-shaped ones count."""
    stripped = text.strip().strip('"')
    if not stripped or " " in stripped:
        return None
    return offending_token(stripped)


def json_pointer(parts: tuple[str, ...]) -> str:
    """RFC 6901 pointer.  Unambiguous where a dotted trail is not: registry
    route ids such as ``tools.hrrr_single_domain_benchmark`` contain dots."""
    if not parts:
        return ""
    escaped = (p.replace("~", "~0").replace("/", "~1") for p in parts)
    return "/" + "/".join(escaped)


class Violation:
    __slots__ = ("path", "line", "kind", "text", "position", "detail")

    def __init__(
        self, path: str, line: int, kind: str, text: str, position: str,
        detail: str = "",
    ) -> None:
        self.path = path
        self.line = line
        self.kind = kind
        self.text = text
        self.position = position
        #: Why this occurrence offends when the spelling alone does not
        #: say so (an import-borne leak names the token in the imported
        #: value).  Deliberately NOT part of the suppression key: the key
        #: must stay stable when the imported module is reworded.
        self.detail = detail

    @property
    def key(self) -> str:
        """Position-aware suppression key.

        The position is part of the key on purpose.  Without it one baseline
        entry suppresses a token everywhere in a file, which is how a physics
        template named after a data source slips past a gate whose whole job
        is to catch physics templates named after data sources.
        """
        return f"{self.path}::{self.position}::{self.kind}::{self.text}"

    def __str__(self) -> str:
        where = f"{self.path}:{self.line}" if self.line else self.path
        because = f" ({self.detail})" if self.detail else ""
        return (
            f"{where}: {self.kind} {self.text!r} at {self.position} "
            f"carries a case token{because}"
        )


def _in_protected_zone(rel: str) -> bool:
    return any(
        rel == zone or (zone.endswith("/") and rel.startswith(zone))
        for zone in PROTECTED_ZONES
    )


def _docstring_ids(tree: ast.AST) -> set[int]:
    marked: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
            if isinstance(first.value.value, str):
                marked.add(id(first.value))
    return marked


# path -> (stat signature, parsed module).  Keyed on the signature so a
# rewritten file is re-parsed: a scan that answered from a stale AST would
# report a violation the file no longer has, or miss one it just gained.
_AST_CACHE: dict[Path, tuple[tuple[int, int], ast.Module | None]] = {}
#: How many re-export hops to follow from an import to the binding that
#: actually holds the value.  A re-export chain longer than this is not a
#: thing this codebase has, and an unbounded walk would be a cycle risk.
_MAX_REEXPORT_HOPS = 4


def _parse_cached(path: Path) -> ast.Module | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    signature = (stat.st_size, stat.st_mtime_ns)
    cached = _AST_CACHE.get(path)
    if cached is not None and cached[0] == signature:
        return cached[1]
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, SyntaxError):
        tree = None
    _AST_CACHE[path] = (signature, tree)
    return tree


def _module_file(root: Path, dotted: str) -> Path | None:
    """Resolve a dotted module name to a file inside the scanned root."""
    if not dotted:
        return None
    base = root.joinpath(*dotted.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _absolute_module(rel: str, module: str | None, level: int) -> str | None:
    """Dotted name of an import target, resolving ``from . import x``."""
    if level == 0:
        return module
    parts = rel.split("/")[:-1]
    if level - 1 > len(parts):
        return None
    prefix = parts[: len(parts) - (level - 1)]
    return ".".join([*prefix, *(module.split(".") if module else [])]) or None


def _value_offence(node: ast.AST) -> str | None:
    """The case token inside an expression, if any."""
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            hit = offending_token(child.id)
        elif isinstance(child, ast.Attribute):
            hit = offending_token(child.attr)
        elif isinstance(child, ast.Constant) and isinstance(child.value, str):
            hit = literal_offence(child.value)
        else:
            continue
        if hit is not None:
            return hit
    return None


def _bound_value_offence(
    root: Path, path: Path, name: str, seen: set[tuple[Path, str]], depth: int
) -> str | None:
    """Case token in the value ``name`` is bound to at ``path`` module scope.

    Functions and classes are deliberately not followed: the rule this gate
    enforces is about case-specialized *data* reaching generic code, and
    treating every imported helper as data would drown the signal.
    """
    if depth > _MAX_REEXPORT_HOPS or (path, name) in seen:
        return None
    seen.add((path, name))
    tree = _parse_cached(path)
    if tree is None:
        return None
    rel = path.relative_to(root).as_posix()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = [t for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target,
                                                            ast.Name):
            targets = [node.target]
        else:
            targets = []
        if any(target.id == name for target in targets):
            if node.value is None:
                return None
            hit = _value_offence(node.value)
            if hit is not None:
                return hit
            # A bare alias (``PUBLIC = _private``) forwards to whatever the
            # aliased name is bound to in the same module.
            if isinstance(node.value, ast.Name):
                return _bound_value_offence(
                    root, path, node.value.id, seen, depth + 1)
            return None
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if (alias.asname or alias.name) != name:
                    continue
                dotted = _absolute_module(rel, node.module, node.level)
                target = None if dotted is None else _module_file(root, dotted)
                if target is not None:
                    return _bound_value_offence(
                        root, target, alias.name, seen, depth + 1)
    return None


def _import_offences(
    node: ast.Import | ast.ImportFrom, root: Path | None, rel: str
) -> list[tuple[str, str, str]]:
    """``(kind, text, detail)`` for one import statement.

    Two independent checks: the spelling of the module/name itself, and --
    when the target resolves inside the scanned root -- the value the
    imported name is bound to over there.  The second is the one a
    per-file scanner structurally cannot see.
    """
    found: list[tuple[str, str, str]] = []
    if isinstance(node, ast.Import):
        for alias in node.names:
            spelling = alias.asname or alias.name
            if offending_token(alias.name) is not None or (
                    offending_token(spelling) is not None):
                found.append(("import", alias.name, ""))
        return found

    module = node.module or ""
    dotted = _absolute_module(rel, node.module, node.level)
    module_offends = offending_token(module) is not None
    target = None if (root is None or dotted is None) else _module_file(
        root, dotted)
    for alias in node.names:
        spelling = alias.asname or alias.name
        label = f"{module}.{alias.name}" if module else alias.name
        if (module_offends or offending_token(alias.name) is not None
                or offending_token(spelling) is not None):
            found.append(("import", label, ""))
            continue
        if target is None or alias.name == "*":
            continue
        token = _bound_value_offence(root, target, alias.name, set(), 0)
        if token is not None:
            found.append((
                "imported value", label,
                f"{dotted}.{alias.name} is bound to a value carrying "
                f"{token!r}"))
    return found


def _ordinals(raw: list[tuple[int, int, str, str, str]]) -> list[str]:
    """Position each occurrence by its ordinal within its (scope, kind, text) group.

    ``raw`` must already be in source order, so the numbering does not depend
    on AST traversal order.  Ordinals replace line numbers because an edit
    elsewhere in the file must not invalidate the baseline -- but they still
    move when an occurrence is *added*, which is the case that matters.
    """
    seen: dict[tuple[str, str, str], int] = {}
    out: list[str] = []
    for item in raw:
        _line, _col, kind, text, scope = item[:5]
        group = (scope, kind, text)
        seen[group] = seen.get(group, 0) + 1
        out.append(f"{scope}#{seen[group]}")
    return out


def scan_python(path: Path, rel: str, root: Path | None = None
                ) -> list[Violation]:
    tree = _parse_cached(path)
    if tree is None:
        return []

    docstrings = _docstring_ids(tree)
    # (line, col, kind, text, scope, detail).  ``detail`` rides along but is
    # part of neither the sort key nor the ordinal group, so it can never
    # move a suppression key.
    raw: list[tuple[int, int, str, str, str, str]] = []

    def record(node: ast.AST, kind: str, text: str, scope: str,
               detail: str = "") -> None:
        raw.append(
            (
                getattr(node, "lineno", 0),
                getattr(node, "col_offset", 0),
                kind,
                text,
                scope,
                detail,
            )
        )

    def visit(node: ast.AST, scope: str) -> None:
        for child in ast.iter_child_nodes(node):
            inner = scope
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                # An import binds a name from ANOTHER file, so neither the
                # module path nor the value behind the name is visible to
                # the identifier/literal walk below.
                for kind, text, detail in _import_offences(child, root, rel):
                    record(child, kind, text, scope, detail)
            elif isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                # The name is bound in the *enclosing* scope; the body is not.
                if offending_token(child.name) is not None:
                    record(child, "definition", child.name, scope)
                inner = f"{scope}.{child.name}"
            elif isinstance(child, ast.Name):
                if offending_token(child.id) is not None:
                    record(child, "identifier", child.id, scope)
            elif isinstance(child, ast.arg):
                if offending_token(child.arg) is not None:
                    record(child, "parameter", child.arg, scope)
            elif isinstance(child, ast.Attribute):
                if offending_token(child.attr) is not None:
                    record(child, "attribute", child.attr, scope)
            elif isinstance(child, ast.Constant) and isinstance(child.value, str):
                if (
                    id(child) not in docstrings
                    and literal_offence(child.value) is not None
                ):
                    record(child, "string literal", child.value.strip(), scope)
            visit(child, inner)

    visit(tree, MODULE_SCOPE)
    raw.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    return [
        Violation(rel, line, kind, text, position, detail)
        for (line, _col, kind, text, _scope, detail), position
        in zip(raw, _ordinals(raw))
    ]


def _strip_rust_comments(source: str) -> str:
    out: list[str] = []
    i, n = 0, len(source)
    line_comment = block_comment = in_string = False
    while i < n:
        ch = source[i]
        nxt = source[i + 1] if i + 1 < n else ""
        if line_comment:
            out.append(ch if ch == "\n" else " ")
            if ch == "\n":
                line_comment = False
        elif block_comment:
            if ch == "*" and nxt == "/":
                block_comment = False
                out.append("  ")
                i += 2
                continue
            out.append("\n" if ch == "\n" else " ")
        elif in_string:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(source[i + 1])
                i += 2
                continue
            if ch == '"':
                in_string = False
        else:
            if ch == "/" and nxt in {"/", "*"}:
                if nxt == "/":
                    line_comment = True
                else:
                    block_comment = True
                out.append("  ")
                i += 2
                continue
            if ch == '"':
                in_string = True
            out.append(ch)
        i += 1
    return "".join(out)


def scan_rust(path: Path, rel: str) -> list[Violation]:
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    found: list[Violation] = []
    # Rust is scanned lexically, so there is no scope to hang a position on;
    # the ordinal is counted over the whole file instead.
    seen: dict[tuple[str, str], int] = {}

    def emit(number: int, kind: str, text: str) -> None:
        group = (kind, text)
        seen[group] = seen.get(group, 0) + 1
        found.append(Violation(rel, number, kind, text, f"#{seen[group]}"))

    for number, line in enumerate(_strip_rust_comments(source).splitlines(), start=1):
        for match in _RUST_STRING_RE.finditer(line):
            text = match.group(0)
            if literal_offence(text) is not None:
                emit(number, "string literal", text.strip('"'))
        for match in _RUST_IDENT_RE.finditer(_RUST_STRING_RE.sub('""', line)):
            ident = match.group(0)
            if offending_token(ident) is not None:
                emit(number, "identifier", ident)
    return found


def scan_json(path: Path, rel: str) -> list[Violation]:
    """Registry documents: every key and scalar string is a physics identifier."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return []

    found: list[Violation] = []

    def walk(node: object, trail: tuple[str, ...]) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                child = trail + (str(key),)
                if offending_token(str(key)) is not None:
                    found.append(
                        Violation(rel, 0, "registry key", str(key), json_pointer(child))
                    )
                walk(value, child)
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, trail + (str(index),))
        elif isinstance(node, str) and literal_offence(node) is not None:
            found.append(
                Violation(rel, 0, "registry value", node, json_pointer(trail))
            )

    walk(document, ())
    return found


def scan_file(path: Path, rel: str, root: Path | None = None
              ) -> list[Violation]:
    """Scan one file.  ``root`` enables import-target resolution; without
    it only the spelling of an import is checked, not the value behind it."""
    if path.suffix == ".py":
        return scan_python(path, rel, root)
    if path.suffix == ".rs":
        return scan_rust(path, rel)
    if path.suffix == ".json":
        return scan_json(path, rel)
    return []


def scan_root(root: Path) -> list[Violation]:
    """Scan every protected zone under ``root``.

    The walk is confined to :data:`PROTECTED_ZONES` rather than run over
    the whole of ``root``, because the zones are the entire scan: a file
    outside them was always discarded, so descending anywhere else buys
    no coverage and costs everything it touches.

    The breakage that names this: walking all of ``root`` reached
    ``.worktrees/Documents``, a link out of the repository into the
    developer's own documents, and stat'd a broken reparse point deep
    inside it.  ``OSError: [WinError 1920]`` came out of ``is_file()``
    BEFORE the zone filter could discard the path, so an unreadable file
    that this gate never intended to read killed the whole scan --
    ``tests/test_case_token_leakage.py`` red on a tree with nothing
    wrong in it, and a gate that cannot run is a gate that is not there.
    Zone membership is a string test on a path, so deciding scope before
    touching the filesystem is what keeps anything outside the zones --
    unreadable, enormous, or merely none of this gate's business -- from
    deciding whether the gate reports.

    Coverage is unchanged: no zone is a prefix of another, so each file
    is reached exactly once, and the set of files scanned is the same
    set the whole-tree walk selected.
    """

    return scan_zones(root, PROTECTED_ZONES)[0]


def scan_zones(root: Path, zones: Sequence[str]
               ) -> tuple[list[Violation], int]:
    """``(violations, files scanned)`` over ``zones`` beneath ``root``.

    The file count is returned rather than inferred because "no violation"
    and "no file" print identically and are opposite verdicts: the CLI
    reported the first for months while doing the second.
    """

    found: list[Violation] = []
    scanned = 0
    for zone in zones:
        target = root / zone
        if not zone.endswith("/"):
            if target.is_file():
                scanned += 1
                found.extend(scan_file(target, zone, root))
            continue
        if not target.is_dir():
            continue
        for path in sorted(target.rglob("*")):
            if path.suffix not in {".py", ".rs", ".json"} or not path.is_file():
                continue
            scanned += 1
            found.extend(scan_file(path, path.relative_to(root).as_posix(), root))
    return found, scanned


def zones_under(prefix: str | None) -> tuple[str, ...]:
    """The protected zones a scan of ``prefix`` covers.

    ``None`` -- the argument WAS the repository root -- is every zone.  A
    prefix that contains zones selects those zones; a prefix that lives
    inside a directory zone selects itself, so pointing the tool at one
    subdirectory of the core scans that subdirectory rather than silently
    widening back to the whole zone or narrowing to nothing.
    """

    if not prefix:
        return PROTECTED_ZONES
    inside = prefix.rstrip("/") + "/"
    covered: list[str] = []
    for zone in PROTECTED_ZONES:
        if zone == prefix.rstrip("/") or zone.startswith(inside):
            covered.append(zone)
        elif zone.endswith("/") and inside.startswith(zone):
            covered.append(inside)
    return tuple(dict.fromkeys(covered))


def _holds_a_zone(directory: Path) -> bool:
    """Whether ``directory`` is a repository root the zones are spelled for."""

    for zone in PROTECTED_ZONES:
        try:
            if (directory / zone).exists():
                return True
        except OSError:
            # Same reason the walk is confined to the zones: an unreadable
            # reparse point on the way past must not decide this answer.
            continue
    return False


def resolve_target(target: Path) -> tuple[Path, str | None] | None:
    """``(repository root, subtree prefix)`` for one CLI argument.

    ``None`` when no directory at or above ``target`` holds a protected
    zone -- there is nothing this gate can say about such a path, and
    saying "clean" about it is the defect this function exists to close.

    The root is the REPOSITORY, never the argument, because every baseline
    key is a repository-relative path: a scan rooted at ``gpuwm`` would
    report ``core/dyn.py`` and match no baseline entry even if it did look
    at the file.
    """

    resolved = Path(target).resolve()
    candidate = resolved if resolved.is_dir() else resolved.parent
    while True:
        if _holds_a_zone(candidate):
            if candidate == resolved:
                return candidate, None
            return candidate, resolved.relative_to(candidate).as_posix()
        if candidate.parent == candidate:
            return None
        candidate = candidate.parent


def target_refusal(target: Path) -> str:
    """Why a target this gate cannot scan is refused, and what to pass."""

    return (
        f"{target}: nothing to scan. No directory at or above it holds any "
        f"of the {len(PROTECTED_ZONES)} protected zones, so this run would "
        "examine zero files -- and a zero-file scan prints exactly what a "
        "clean tree prints, which is how a case token in the numerical core "
        "reads green for months. Pass the repository root, or a path inside "
        "it: python tools/check_case_token_leakage.py .")


def partition(
    violations: list[Violation], baseline: dict[str, str] | None = None
) -> tuple[list[Violation], list[Violation]]:
    """Split into ``(new, suppressed)`` against a position-aware baseline."""
    table = BASELINE if baseline is None else baseline
    new: list[Violation] = []
    suppressed: list[Violation] = []
    for violation in violations:
        (suppressed if violation.key in table else new).append(violation)
    return new, suppressed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots", nargs="+", type=Path, metavar="PATH",
        help="a repository root, or any path inside one; the zones are "
             "spelled against the repository, so the repository is found "
             "by walking up from what you pass")
    parser.add_argument("--show-baseline", action="store_true")
    parser.add_argument(
        "--emit-baseline",
        action="store_true",
        help="print a baseline skeleton for every occurrence found; the "
        "reasons come out as UNREVIEWED and suppress nothing until a human "
        "replaces each one with an actual decision",
    )
    args = parser.parse_args(argv)

    found: list[Violation] = []
    scanned = 0
    zone_count = 0
    for target in args.roots:
        resolved = resolve_target(target)
        if resolved is None:
            print(target_refusal(target), file=sys.stderr)
            return 2
        root, prefix = resolved
        zones = zones_under(prefix)
        violations, files = scan_zones(root, zones)
        found.extend(violations)
        scanned += files
        zone_count += len(zones)

    if scanned == 0:
        print(
            f"{' '.join(str(t) for t in args.roots)}: scanned 0 file(s) "
            f"across {zone_count} selected zone(s). Every protected zone "
            "the arguments selected is absent from this tree, so this run "
            "would report a clean bill of health without reading anything. "
            "Pass the repository root: "
            "python tools/check_case_token_leakage.py .",
            file=sys.stderr)
        return 2

    if args.emit_baseline:
        skeleton = {
            "comment": (
                "Pre-existing case/source token leakage, recorded so the gate "
                "blocks regressions.  Keys are position-aware; every entry "
                "needs a written reason."
            ),
            "known_violations": {
                v.key: f"UNREVIEWED: {v.kind} {v.text!r}" for v in found
            },
        }
        print(json.dumps(skeleton, indent=2, sort_keys=True))
        return 0

    if BASELINE_ERROR is not None:
        print(f"baseline unusable, nothing is suppressed: {BASELINE_ERROR}", file=sys.stderr)

    new, suppressed = partition(found)

    if args.show_baseline and suppressed:
        print(f"suppressed by BASELINE ({len(suppressed)}):")
        for violation in suppressed:
            print(f"  {violation}")
            print(f"      reason: {BASELINE[violation.key]}")
        print()

    if new:
        print(f"case/source token in a generic position ({len(new)} of "
              f"{scanned} file(s) scanned in {zone_count} protected zone(s)):")
        for violation in new:
            print(f"  {violation}")
            print(f"      key: {violation.key}")
        print(
            "\nA case or source token belongs in case data, a source-specific "
            "adapter, or a test fixture.\nMove the behavior into data (a "
            "registry entry or mapping authority) and keep the identifier generic."
        )
        return 1

    debt = sum(1 for v in suppressed if reason_is_debt(BASELINE[v.key]))
    print(
        f"no NEW case/source token in a generic position "
        f"(scanned {scanned} file(s) in {zone_count} protected zone(s); "
        f"{len(suppressed)} known occurrence(s) suppressed; "
        f"{debt} of them carried as unretired violations)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
