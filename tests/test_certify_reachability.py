"""A validator nobody calls is the same defect wherever it lives.

Three functions under ``gpuwm/certify/`` were written, documented, exported
and tested -- and called by nothing.  ``gpuwm certify`` therefore printed
PASS over NVRTC drift and over an empty kernel manifest, because the code
that would have caught both was unreachable from the command.  A test suite
that exercises a validator directly does not notice: the function passes its
own tests forever while the product never runs it.

So this module does not ask "is it tested".  It asks **"is it reachable from
the front door"** -- from ``gpuwm certify`` / ``gpuwm dual-run`` and from the
capsule-emission path -- by walking the package's own call graph.

The instrument is validated in both directions against a synthetic package
whose answers are known, because a reachability checker that silently reports
"all reachable" is exactly the failure it is meant to catch.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from gpuwm.certify import CERTIFICATION_CHECKS

REPO = Path(__file__).resolve().parents[1]
CERTIFY = REPO / "gpuwm" / "certify"

#: Where a reader enters this package.  ``cli`` is the shipped command;
#: ``emit_run_capsule`` is the other front door, the one every finishing
#: forecast goes through.
ENTRY_POINTS: tuple[str, ...] = (
    "cli:register_cli",
    "cli:_certify_main",
    "cli:_dual_run_main",
    "capsule:emit_run_capsule",
)

#: A function whose name says it decides whether something is acceptable.
#: Matched against every module-level function in the package; anything it
#: catches must be declared in :data:`CERTIFICATION_CHECKS`, so a validator
#: added later cannot quietly skip the reachability requirement.
VALIDATOR_NAME = re.compile(
    r"^(validate_|absent_|unresolved_|failing_|describe_drift$|"
    r"recorded_compile_platform$|compile_platform_agreement$)"
    r"|_is_empty$")


# --------------------------------------------------------------------------
# The call graph
# --------------------------------------------------------------------------

def _function_defs(source: str) -> dict[str, ast.AST]:
    tree = ast.parse(source)
    return {node.name: node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _names_used(node: ast.AST) -> set[str]:
    """Every bare name and attribute this function body mentions.

    Attributes count, so ``verdict.certify`` and ``from x import certify``
    are the same edge.  Over-counting is the safe direction here: it can
    only make something look reachable that is referenced but not invoked,
    and a referenced-not-invoked validator is a far smaller defect than an
    unreferenced one -- while UNDER-counting would produce false alarms
    that get the whole test disabled.
    """
    used: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            used.add(child.id)
        elif isinstance(child, ast.Attribute):
            used.add(child.attr)
        elif isinstance(child, ast.ImportFrom):
            used.update(alias.asname or alias.name for alias in child.names)
    return used


def build_call_graph(sources: dict[str, str]) -> dict[str, set[str]]:
    """``{"module:function": {"module:function", ...}}`` over one package.

    An edge exists when the caller's body mentions the callee's NAME and the
    package defines exactly that name.  Cross-module by name, because these
    modules import each other's functions directly.
    """
    defs = {module: _function_defs(text) for module, text in sources.items()}
    owners: dict[str, list[str]] = {}
    for module, functions in defs.items():
        for name in functions:
            owners.setdefault(name, []).append(module)
    graph: dict[str, set[str]] = {}
    for module, functions in defs.items():
        for name, node in functions.items():
            used = _names_used(node)
            graph[f"{module}:{name}"] = {
                f"{owner}:{callee}"
                for callee in used
                for owner in owners.get(callee, ())
            }
    return graph


def reachable_from(graph: dict[str, set[str]],
                   entries: tuple[str, ...]) -> set[str]:
    seen: set[str] = set()
    stack = [entry for entry in entries if entry in graph]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(graph.get(node, ()) - seen)
    return seen


def _package_sources() -> dict[str, str]:
    """Every module of the package, by stem.

    ``utf-8-sig`` rather than ``utf-8``, and a named refusal rather than a
    bare ``SyntaxError``, because both failure modes are ones this test
    already produced on itself: a tool that rewrote one module with a BOM
    made ``ast.parse`` raise INSIDE the test body, which pytest reports as
    a reachability FAILURE on every declared check at once.  A guard whose
    loudest failure is a lie about what it measures is the defect it exists
    to catch, one level up.
    """
    sources = {}
    for path in sorted(CERTIFY.glob("*.py")):
        text = path.read_text(encoding="utf-8-sig")
        try:
            ast.parse(text)
        except SyntaxError as error:
            raise AssertionError(
                f"{path} does not parse ({error}); this is a broken file, "
                f"NOT an unreachable validator") from error
        sources[path.stem] = text
    return sources


# --------------------------------------------------------------------------
# Validating the instrument -- both directions, on known answers
# --------------------------------------------------------------------------

_DEAD = {
    "cli": "def main():\n    return helper()\n",
    "lib": ("def helper():\n    return 1\n\n\n"
            "def validate_thing(x):\n    return bool(x)\n"),
}

_WIRED = {
    "cli": "def main():\n    return helper()\n",
    "lib": ("def helper():\n    return validate_thing(1)\n\n\n"
            "def validate_thing(x):\n    return bool(x)\n"),
}


def test_the_detector_finds_a_validator_nobody_calls():
    """Direction 1: an unreachable validator IS reported unreachable."""
    graph = build_call_graph(_DEAD)
    reached = reachable_from(graph, ("cli:main",))
    assert "lib:helper" in reached, "the control edge itself is broken"
    assert "lib:validate_thing" not in reached


def test_the_detector_clears_a_validator_that_is_called():
    """Direction 2: adding the one call makes it reachable.

    Without this, a detector that reported EVERYTHING unreachable would
    pass the test above and be useless.
    """
    graph = build_call_graph(_WIRED)
    reached = reachable_from(graph, ("cli:main",))
    assert "lib:validate_thing" in reached


def test_the_detector_follows_a_chain_rather_than_one_hop():
    """Reachability is transitive, or a validator two calls deep reads dead."""
    sources = {
        "cli": "def main():\n    return one()\n",
        "lib": ("def one():\n    return two()\n\n\n"
                "def two():\n    return validate_deep()\n\n\n"
                "def validate_deep():\n    return True\n"),
    }
    reached = reachable_from(build_call_graph(sources), ("cli:main",))
    assert "lib:validate_deep" in reached


# --------------------------------------------------------------------------
# The real package
# --------------------------------------------------------------------------

def test_the_entry_points_exist():
    """Anti-vacuity: a typo in an entry point would clear every check."""
    graph = build_call_graph(_package_sources())
    for entry in ENTRY_POINTS:
        assert entry in graph, f"{entry} is not a function in gpuwm/certify"


def test_the_declared_check_list_is_not_empty_and_names_real_functions():
    sources = _package_sources()
    defs = {module: set(_function_defs(text))
            for module, text in sources.items()}
    assert CERTIFICATION_CHECKS, "no certification check is declared"
    for qualified in CERTIFICATION_CHECKS:
        module, _, name = qualified.partition(":")
        assert module in defs, qualified
        assert name in defs[module], (
            f"{qualified} is declared as a certification check but "
            f"gpuwm/certify/{module}.py defines no such function")


@pytest.mark.parametrize("qualified", CERTIFICATION_CHECKS)
def test_every_declared_check_is_reachable_from_a_front_door(qualified):
    """THE GUARD.  A check the command cannot reach is a check that is off.

    This is the test that would have been red for
    ``compile_platform_fingerprint``, ``describe_drift`` and
    ``manifest_is_empty`` on 2.3.3 and every release before it.
    """
    graph = build_call_graph(_package_sources())
    reached = reachable_from(graph, ENTRY_POINTS)
    assert qualified in reached, (
        f"{qualified} is declared a certification check but nothing reaches "
        f"it from {list(ENTRY_POINTS)} -- it cannot refuse anything, so "
        f"gpuwm certify would pass over whatever it was written to catch")


def test_every_validator_shaped_function_is_declared_as_a_check():
    """The anti-forgetting leg: a new validator cannot skip the guard.

    Without this, the guard only protects the checks somebody remembered to
    list, which is the same hole one level up.
    """
    undeclared = []
    for module, text in _package_sources().items():
        for name in _function_defs(text):
            if not VALIDATOR_NAME.search(name):
                continue
            if f"{module}:{name}" not in CERTIFICATION_CHECKS:
                undeclared.append(f"{module}:{name}")
    assert undeclared == [], (
        f"validator-shaped functions absent from CERTIFICATION_CHECKS: "
        f"{undeclared} -- declare them, or rename them if they do not decide "
        f"whether something is acceptable")


def test_the_validator_name_rule_is_not_vacuous():
    """The rule must actually match the names it exists to catch."""
    for name in ("validate_band", "describe_drift", "manifest_is_empty",
                 "absent_reference_hashes", "unresolved_pins"):
        assert VALIDATOR_NAME.search(name), name
    for name in ("read_metrics_rows", "canonical_digest", "encode_key"):
        assert not VALIDATOR_NAME.search(name), name


def test_no_public_function_under_certify_is_wholly_unreferenced():
    """The broader sweep that found the two siblings the audit missed.

    ``recorded_module_keys`` and ``load_wrf_reference_manifest`` had zero
    references anywhere in the tree -- product, tools or tests.  Nothing was
    watching for that, so nothing said so.
    """
    sources = _package_sources()
    searched = [REPO / "gpuwm", REPO / "tools", REPO / "tilestream",
                REPO / "tests"]
    referenced: set[str] = set()
    for root in searched:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if "vendor" in path.parts:
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            module = ast.Module(body=list(tree.body), type_ignores=[])
            referenced |= _names_used(module)

    orphans = []
    for module, text in sources.items():
        own = _function_defs(text)
        for name in own:
            if name.startswith("_"):
                continue
            # A reference from anywhere counts here -- including the
            # module itself, which is why this is the weaker sweep and
            # the reachability test above is the real guard.
            elsewhere = any(name in _names_used(node)
                            for other, other_text in sources.items()
                            for node in _function_defs(other_text).values()
                            if not (other == module and node is own[name]))
            if name in referenced or elsewhere:
                continue
            orphans.append(f"{module}:{name}")
    assert orphans == [], (
        f"public functions under gpuwm/certify referenced by nothing in the "
        f"tree: {orphans}")
