"""A146 source scan: float divisions by a compile-time constant, from source.

NVRTC's NVVM, targeting compute_100 and compute_120 (every Blackwell card)
under ``-ftz=true`` (which CuPy appends to every ``RawModule``), compiles
``x / C`` for a compile-time constant ``C`` as ``x * RN(1/C)``, which is not
IEEE division.  ``__fdiv_rn(x, C)`` is never rewritten, so it is the one
spelling a kernel may use for such a division.  A power-of-two, zero or
infinite divisor has an exact reciprocal and either spelling is IEEE.

This module reads source and imports neither CuPy nor NVRTC, so the gate
built on it (tests/test_literal_division_source_gate.py) runs on every leg,
including the CPU legs that set ``GPUWM_NO_LOCAL_GPU``.  It sees a divisor
that is

* a float literal (``x / 3.0f``, ``y /= 1.718e-5f``);
* a name whose value is a float literal: a ``#define`` or ``const`` /
  ``constexpr`` float of the file itself, of ``CUDA_DEFINES`` or of a header
  the loader prepends to the unit (``/ THOMPSON_AA_D0C``);
* a parenthesized constant expression (``/ (THOMPSON_AA_D0R * 2.0f)``);
* a C math function of constant arguments (``/ tgammaf(4.0f)``), which
  compute_120 constant-folds and compute_89 does not.

A divisor that reaches a division through a local variable, an integer
literal or a helper inlined with a literal argument is invisible here; the
compiler census (tools/literal_division_census.py, which needs NVRTC) sees
those.
"""
from __future__ import annotations

import ast
import math
import re
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KDIR = ROOT / "gpuwm" / "core" / "kernels"

#: Kernel sources compiled only through ``compile_using_nvrtc`` with
#: ``--ftz=false`` (the RRTMG routes): the rewrite needs ``-ftz=true``, so a
#: constant division there is still ``div.rn``.  A file moving onto a
#: ``-ftz=true`` route must leave this list.
FTZ_FALSE_SOURCES = frozenset({
    "rrtmg_sw.cu", "rrtmg_lw_chain.cu", "rrtmg_lw_chain_coalesced.cu",
    "rrtmg_lw_taugb02_10_11_12.cu", "rrtmg_lw_taugb03_05.cu",
    "rrtmg_lw_taugb06_09.cu", "rrtmg_lw_taugb13_16.cu",
    "rrtmg_lw_zbatched.cu",
})

_NUMBER = r"(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
_FLOAT_LITERAL = re.compile(rf"(?:{_NUMBER})[fF]\b|\d+[eE][-+]?\d+[fF]\b")
_IDENT = re.compile(r"[A-Za-z_]\w*")

#: Float C math functions NVVM may constant-fold for a constant argument
#: (compute_120 folds ``tgammaf(4.0f)``, compute_89 does not), with the host
#: function that gives their value closely enough to tell a power of two.
_MATH = {
    "tgammaf": math.gamma, "lgammaf": math.lgamma, "sqrtf": math.sqrt,
    "cbrtf": lambda v: math.copysign(abs(v) ** (1.0 / 3.0), v),
    "expf": math.exp, "exp2f": lambda v: 2.0 ** v,
    "exp10f": lambda v: 10.0 ** v, "logf": math.log, "log10f": math.log10,
    "log2f": math.log2, "powf": math.pow, "sinf": math.sin,
    "cosf": math.cos, "tanf": math.tan, "atanf": math.atan, "fabsf": abs,
}
#: A double literal (no ``f`` suffix): an expression holding one is double,
#: and a double division by a constant is IEEE on every card measured.
_DOUBLE_LITERAL = re.compile(
    r"(?<![\w.])(?:\d+\.\d*|\.\d+|\d+[eE][-+]?\d+)(?:[eE][-+]?\d+)?(?![\w.])")


def code_only(text: str) -> str:
    """Comments and string/char literals blanked, offsets kept."""
    out = list(text)
    i, n = 0, len(text)
    while i < n:
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
        elif text[i] in "\"'":
            q, j = text[i], i + 1
            while j < n and text[j] != q:
                j += 2 if text[j] == "\\" else 1
            j += 1
        else:
            i += 1
            continue
        for k in range(i, min(j, n)):
            if out[k] != "\n":
                out[k] = " "
        i = j
    return "".join(out)


_CONST_CHARS = frozenset(
    "0123456789.eEfF+-*/() \t\n_"
    "abcdghijklmnopqrstuvwxyzABCDGHIJKLMNOPQRSTUVWXYZ")


def _is_constant_expr(expr: str, names) -> bool:
    """Numbers, known constant names, arithmetic and casts only."""
    expr = expr.strip()
    if not expr or not set(expr) <= _CONST_CHARS:
        return False
    bare = re.sub(r"\((?:float|double|real)\)", " ", expr)
    bare = re.sub(rf"(?<![\w.]){_NUMBER}[fF]?", " ", bare)
    return all(tok in names for tok in _IDENT.findall(bare))


def _float_constants(code: str, extra_names=(), extra_values=None
                     ) -> tuple[dict[str, float], set[str]]:
    """Float constants with a known value, and every compile-time constant
    name (a ``#define`` or ``const``/``constexpr`` whose initializer is a
    constant expression).

    The scan is not scope-aware, so a declared name only gets a value when
    every float declaration of it in the file (const or not, parameters
    included) is a constant expression of one value: ``am_r`` is
    ``3.1415926536f * 1000.0f / 6.0f`` wherever it is declared, while
    ``rho`` is a constant in one kernel and a runtime density in forty."""
    from gpuwm.core.constants import CUDA_DEFINES
    values = {k: float(np.float32(v)) for k, v in CUDA_DEFINES.items()}
    values.update(extra_values or {})
    for m in re.finditer(
            rf"^\s*#\s*define\s+([A-Za-z_]\w*)\s+\(?\s*(-?{_NUMBER}[fF])\s*\)?\s*$",
            code, re.M):
        values[m.group(1)] = float(np.float32(m.group(2).rstrip("fF")))
    fixed = set(values)
    names = set(values) | set(extra_names)
    defines = re.findall(r"^\s*#\s*define\s+([A-Za-z_]\w*)[ \t]+(.+)$", code, re.M)
    decls = []
    for m in re.finditer(
            r"(?:const|constexpr)\s+(?:static\s+)?(?:__device__\s+)?"
            r"(?:float|real|double|int)\s+([^;{}()]+(?:\([^;{}]*\))?[^;{}]*);",
            code):
        # ``const real a = 1.0f, b = a / 2.0f;`` declares several names.
        for part in m.group(1).split(","):
            name, eq, expr = part.partition("=")
            if eq and _IDENT.fullmatch(name.strip()):
                decls.append((name.strip(), expr))
    # Every float declaration of each name: its initializer when it is a
    # const one, else None (a mutable variable, a parameter, a pointer or an
    # uninitialized declaration).
    declared: dict[str, list] = {}
    for m in re.finditer(r"(?:\b(const|constexpr)\s+(?:static\s+)?"
                         r"(?:__device__\s+)?)?(?<![\w.])(?:float|real)\s+"
                         r"([A-Za-z_]\w*)\s*(?:=\s*([^;{}]+?))?\s*[;,)]",
                         code):
        declared.setdefault(m.group(2), []).append(
            m.group(3) if m.group(1) else None)
    for m in re.finditer(r"(?<![\w.])(?:float|real)\s*[*&]\s*([A-Za-z_]\w*)",
                         code):
        declared.setdefault(m.group(1), []).append(None)
    for _ in range(4):
        for name, expr in defines + decls:
            if name not in names and _is_constant_expr(expr, names):
                names.add(name)
        for name, exprs in declared.items():
            if name in fixed or name in values or None in exprs:
                continue
            found = set()
            for expr in exprs:
                if _DOUBLE_LITERAL.search(expr) or not _is_constant_expr(
                        expr, names):
                    found = None
                    break
                value = _evaluate(expr, values)
                if value is None:
                    found = None
                    break
                found.add(float(np.float32(value)))
            if found is not None and len(found) == 1:
                values[name] = found.pop()
    return values, names


def _exact(value: float) -> bool:
    """A power of two, zero or infinity: its reciprocal is exact."""
    bits = struct.unpack("<I", struct.pack("<f", np.float32(value)))[0]
    return (bits & 0x007FFFFF) == 0


def _match_back(code: str, close: int) -> int:
    depth, k = 0, close
    while k >= 0:
        if code[k] == ")":
            depth += 1
        elif code[k] == "(":
            depth -= 1
            if depth == 0:
                return k
        k -= 1
    return -1


def _match_forward(code: str, open_: int) -> int:
    depth = 0
    for k in range(open_, len(code)):
        if code[k] == "(":
            depth += 1
        elif code[k] == ")":
            depth -= 1
            if depth == 0:
                return k
    return -1


def _numerator_is_constant(code: str, slash: int, names) -> bool:
    """True when the whole multiplicative left operand is a constant
    expression, as in ``PI * 1000.0f / 6.0f``: the compiler folds it."""
    j = slash - 1
    while True:
        while j >= 0 and code[j].isspace():
            j -= 1
        if j < 0:
            return False
        if code[j] == ")":
            k = _match_back(code, j)
            if k < 0 or not _is_constant_expr(code[k + 1:j], names):
                return False
            p = k - 1
            while p >= 0 and code[p].isspace():
                p -= 1
            if p >= 0 and (code[p].isalnum() or code[p] == "_"):
                return False          # a call f(...), not a group
            j = k - 1
        else:
            m = re.search(rf"(?:{_NUMBER}[fF]?|[A-Za-z_]\w*)$", code[:j + 1])
            if not m:
                return False
            tok = m.group(0)
            if not (re.fullmatch(rf"{_NUMBER}[fF]?", tok) or tok in names):
                return False
            j = m.start() - 1
        while j >= 0 and code[j].isspace():
            j -= 1
        if j >= 0 and code[j] in "*/%" and not (j and code[j - 1] in "*/"):
            j -= 1
            continue
        if j >= 0 and code[j] in "-+" and (j == 0 or code[j - 1] in "(,=?:*/"):
            j -= 1              # unary sign on the leading constant
            while j >= 0 and code[j].isspace():
                j -= 1
        return j < 0 or code[j] in "(,=?:+-<>&|!;{}" or code[j] == "\n"


def _evaluate(expr: str, values: dict[str, float]):
    """The value of a constant C expression, or None when it cannot be
    evaluated here (an unknown name, an integer-only expression)."""
    text = re.sub(r"\((?:float|double|real)\)", " ", expr)
    if not (_FLOAT_LITERAL.search(text) or re.search(r"\d\.\d*|\.\d", text)
            or any(name in values for name in _IDENT.findall(text))):
        return None      # integer arithmetic: the census's job, not ours
    text = re.sub(rf"(?<![\w.])({_NUMBER})[fF]\b", r"\1", text)
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in values \
                and node.id not in _MATH:
            return None
        if not isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp,
                                 ast.Constant, ast.Name, ast.Load, ast.Call,
                                 ast.Add, ast.Sub, ast.Mult, ast.Div,
                                 ast.USub, ast.UAdd)):
            return None
    try:
        return float(eval(compile(tree, "<divisor>", "eval"),  # noqa: S307
                          {"__builtins__": {}}, {**_MATH, **values}))
    except (ArithmeticError, ValueError, TypeError):
        return None


def _divisor(code: str, start: int, values, names):
    """(token, value) of a float constant divisor starting at ``start``,
    else None."""
    rest = code[start:]
    lit = _FLOAT_LITERAL.match(rest)
    if lit:
        return lit.group(0), float(np.float32(lit.group(0).rstrip("fF")))
    ident = _IDENT.match(rest)
    if ident:
        name = ident.group(0)
        after = rest[ident.end():]
        stripped = after.lstrip()
        if name in _MATH and stripped.startswith("("):
            open_ = start + ident.end() + (len(after) - len(stripped))
            close = _match_forward(code, open_)
            if close < 0:
                return None
            args = code[open_ + 1:close]
            if _DOUBLE_LITERAL.search(args) or not all(
                    _is_constant_expr(a, names) for a in args.split(",")):
                return None
            value = _evaluate(code[start:close + 1], values)
            if value is None:
                return None   # not evaluable here: the census's job
            return re.sub(r"\s+", " ", code[start:close + 1]), value
        if name not in values or stripped.startswith(("(", "[", ".", "->")):
            return None
        return name, values[name]
    if rest.startswith("("):
        close = _match_forward(code, start)
        if close < 0:
            return None
        inner = code[start + 1:close]
        if inner.strip() in ("float", "double", "real") or not inner.strip():
            return None       # a cast: ``x / (float)n``
        if _DOUBLE_LITERAL.search(inner) or not _is_constant_expr(inner,
                                                                  names):
            return None
        value = _evaluate(inner, values)
        if value is None:
            return None       # an integer or unevaluable constant: the census
        return "(" + re.sub(r"\s+", " ", inner.strip()) + ")", value
    return None


def literal_divisions(text: str, extra_names=(), extra_values=None
                      ) -> list[tuple[int, str]]:
    """(line, divisor) of every plain ``/`` by a float constant.

    ``extra_names`` and ``extra_values`` are the constants the translation
    unit receives from a header prepended to it (common.cuh, an extra
    header)."""
    code = code_only(text)
    values, names = _float_constants(code, extra_names, extra_values)
    hits = []
    for m in re.finditer(r"/(=?)\s*", code):
        s = m.start()
        if code[s + 1:s + 2] in ("/", "*") or (s and code[s - 1] in "*/"):
            continue
        found = _divisor(code, m.end(), values, names)
        if found is None:
            continue
        token, value = found
        if _exact(value):
            continue
        if not m.group(1) and _numerator_is_constant(code, s, names):
            continue
        hits.append((code.count("\n", 0, s) + 1, token))
    return hits


def header_constants(path: Path) -> tuple[set[str], dict[str, float]]:
    """Constant names and float values the loader prepends to ``path``'s
    unit (common.cuh and the unit's extra headers)."""
    from gpuwm.core.kernels import EXTRA_HEADERS
    headers = ["common.cuh"] + list(EXTRA_HEADERS.get(path.stem, ()))
    names: set[str] = set()
    values: dict[str, float] = {}
    for header in headers:
        if header != path.name:
            code = code_only((KDIR / header).read_text(encoding="utf-8"))
            hv, hn = _float_constants(code)
            names |= hn
            values.update(hv)
    return names, values


def ftz_true_sources() -> list[Path]:
    return sorted(p for p in list(KDIR.glob("*.cu")) + list(KDIR.glob("*.cuh"))
                  if p.name not in FTZ_FALSE_SOURCES)


def kernel_file_offenders() -> list[str]:
    """``path:line divides by C`` for every -ftz=true kernel source."""
    offenders = []
    for path in ftz_true_sources():
        names, values = header_constants(path)
        for line, token in literal_divisions(
                path.read_text(encoding="utf-8"), names, values):
            offenders.append(f"{path.relative_to(ROOT).as_posix()}:{line} "
                             f"divides by {token}")
    return offenders


_KERNEL_FACTORIES = ("ElementwiseKernel", "ReductionKernel")


def inline_kernel_strings(source: str):
    """(line, text) of every CUDA source string in a Python module: string
    constants holding ``__global__`` or ``__device__`` code, and the
    operation strings handed to CuPy's ElementwiseKernel/ReductionKernel."""
    tree = ast.parse(source)
    seen = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(
                func, "id", "")
            if name in _KERNEL_FACTORIES:
                for arg in list(node.args) + [k.value for k in node.keywords]:
                    for sub in ast.walk(arg):
                        if isinstance(sub, ast.Constant) and isinstance(
                                sub.value, str) and id(sub) not in seen:
                            seen.add(id(sub))
                            yield sub.lineno, sub.value
        elif (isinstance(node, ast.Constant) and isinstance(node.value, str)
              and ("__global__" in node.value or "__device__" in node.value)
              and id(node) not in seen):
            seen.add(id(node))
            yield node.lineno, node.value


def inline_kernel_offenders(root: Path = ROOT) -> list[str]:
    """The same scan over the CUDA source strings of every ``gpuwm`` module."""
    common_names, common_values = header_constants(KDIR / "common.cu")
    offenders = []
    for path in sorted((root / "gpuwm").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "__global__" not in text and "__device__" not in text \
                and not any(f in text for f in _KERNEL_FACTORIES):
            continue
        for lineno, kernel in inline_kernel_strings(text):
            for line, token in literal_divisions(kernel, common_names,
                                                 common_values):
                offenders.append(
                    f"{path.relative_to(root).as_posix()}:{lineno} "
                    f"(string line {line}) divides by {token}")
    return offenders
