"""Hand-built launch tuples must match the kernel signature's arity.

CLASS 2, in docs/ntiedtke/STANDING-RULES.md section 1's sense: WRF cannot
disagree with a launch tuple.  There is no Fortran analogue to grade it
against, so parity is structurally blind and ``max_ulp == 0`` passes no
matter how wrong the tuple is.  The rule says such a thing needs its own
gate; this is that gate.

It exists because 49a3f357 added ``int tiedtke_closure`` to
``ntiedtke_mfub`` and ``ntiedtke_closure`` immediately before ``ncol``,
and the two hand-built launches in test_ntiedtke_prep_parity.py kept
their old tuples.  Every argument after the insertion point shifted:
``nz`` was read from ``dt``.  CuPy does NOT check arity against the
compiled signature -- there is no ``num_args`` on a RawKernel -- so the
launch ACCESS-VIOLATED and took the whole pytest process down with a
Windows fatal exception rather than failing one test.  The forecast was
bitwise-exact throughout, because the production path builds its
arguments from the signature tables in ntiedtke.py; only a caller that
writes the tuple by hand could break, and only the tests do that.

So: a bitwise forecast gate says nothing about a kernel the forecast
never launches this way, and that is precisely the blindness section 1
is about.
"""
from __future__ import annotations

import ast
import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
KERNEL = REPO / "gpuwm" / "core" / "kernels" / "ntiedtke.cu"
TESTS = sorted((REPO / "tests").glob("test_ntiedtke*.py"))

#: ``extern "C" __global__ void NAME( ... )`` up to the closing paren.
_SIG = re.compile(
    r'extern\s+"C"\s+__global__\s+void\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)\s*\{',
    re.S)


def _count_params(blob: str) -> int:
    """Parameters in a C parameter list, comments and newlines removed."""
    blob = re.sub(r"/\*.*?\*/", " ", blob, flags=re.S)
    blob = re.sub(r"//[^\n]*", " ", blob)
    blob = blob.strip()
    if not blob or blob == "void":
        return 0
    # No templates or function pointers in these signatures, so top-level
    # commas are the separators.
    return len([p for p in blob.split(",") if p.strip()])


def kernel_arities() -> dict[str, int]:
    src = KERNEL.read_text(encoding="utf-8")
    return {m.group(1): _count_params(m.group(2)) for m in _SIG.finditer(src)}


def tail_length() -> int:
    """How many arguments ``NtStages.launch`` appends on the caller's behalf.

    ``_tail()`` supplies the geometry and ordering carriers at the launcher
    rather than at each call site, precisely so a stage cannot opt out of
    those checks -- so a hand-built tuple is SHORTER than the signature by
    exactly this many.  Read from the source instead of hardcoded: the
    first draft of this gate assumed zero and failed all twenty sites at a
    HEAD where the real tests passed, which is the gate being wrong rather
    than the tests.  Hardcoding 5 would fail the same way the day someone
    adds a sixth carrier.
    """
    src = (REPO / "gpuwm" / "core" / "ntiedtke.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_tail":
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.Return) and isinstance(
                        stmt.value, (ast.Tuple, ast.List)):
                    return len(stmt.value.elts)
    raise AssertionError(
        "NtStages._tail was not found, or no longer returns a tuple "
        "display; this gate cannot say how many arguments the launcher "
        "appends and must not silently assume a number.")


def launch_sites() -> list[tuple[str, str, int, int]]:
    """``(file, kernel, lineno, tuple_length)`` for every literal launch.

    Only calls whose first argument is a string literal and whose second
    is a tuple/list DISPLAY are counted -- anything assembled at run time
    cannot be checked statically and is skipped rather than guessed at.
    """
    out = []
    for path in TESTS:
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if not (isinstance(fn, ast.Attribute) and fn.attr == "launch"):
                continue
            if len(node.args) < 2:
                continue
            name, args = node.args[0], node.args[1]
            if not (isinstance(name, ast.Constant)
                    and isinstance(name.value, str)):
                continue
            if not isinstance(args, (ast.Tuple, ast.List)):
                continue
            if any(isinstance(e, ast.Starred) for e in args.elts):
                continue
            out.append((path.name, name.value, node.lineno, len(args.elts)))
    return out


def test_the_gate_sees_a_corpus():
    """A gate that matches nothing passes for the wrong reason.

    Both halves are asserted non-empty, and the two are asserted to
    intersect: signatures alone would pass on a test file that launches
    nothing, and sites alone would pass against an empty kernel file.
    """
    arities = kernel_arities()
    sites = launch_sites()
    assert len(arities) >= 15, f"only {len(arities)} kernel signatures parsed"
    assert len(sites) >= 5, f"only {len(sites)} literal launch sites found"
    named = {k for _f, k, _l, _n in sites}
    assert named & set(arities), (
        "no launch site names a kernel this file can find a signature for; "
        "the comparison below would be vacuous")


@pytest.mark.parametrize(
    "fname,kernel,lineno,ntuple",
    [pytest.param(*s, id=f"{s[0]}:{s[2]}:{s[1]}") for s in launch_sites()])
def test_launch_tuple_matches_signature(fname, kernel, lineno, ntuple):
    arities = kernel_arities()
    if kernel not in arities:
        pytest.skip(f"{kernel} is not declared in ntiedtke.cu")
    # The call site supplies everything EXCEPT the launcher's own tail.
    expected = arities[kernel] - tail_length()
    assert ntuple == expected, (
        f"{fname}:{lineno} launches {kernel} with {ntuple} arguments but the "
        f"kernel takes {expected}.  CuPy does not check this -- a mismatched "
        f"tuple access-violates at launch instead of raising, so the symptom "
        f"is a dead pytest process, not a failed assertion.  If a parameter "
        f"was just added to {kernel}, every hand-built launch of it needs "
        f"the same argument in the same position.")
