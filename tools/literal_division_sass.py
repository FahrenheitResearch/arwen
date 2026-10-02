"""A146: per-kernel SASS digests of every census unit, for one architecture.

The fix is proven not to move an older card by comparing these digests for
sm_89 between two trees: ``x / C`` and ``__fdiv_rn(x, C)`` are both
``div.rn`` there, and ptxas propagates the constant into the same division
sequence.  Two digests per kernel: ``sequence`` (the instruction stream with
addresses, encodings and internal subroutine numbering dropped) and
``multiset`` (the same instructions with register names erased, sorted),
because an immediate divisor and a register holding it can leave the
allocator numbering and ordering the same instructions differently.

    CUDA_VISIBLE_DEVICES= python -m tools.literal_division_sass --arch sm_89 \
        --nvdisasm /path/to/nvdisasm --json out.json [--root tree]

``--root`` names the tree whose sources are compiled (default: the tree
holding this tool), as in tools/literal_division_census.py (A193).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from tools.literal_division_census import ROOT, production_units

_INSN = re.compile(r"^\s+/\*[0-9a-f]{4,}\*/\s+(.*?)\s*(?:/\*.*)?$")


def cubin(source: str, options: tuple[str, ...]) -> bytes:
    # Keyed by its options (A160): NVRTC's cache ignores -ftz for a fixed
    # program name, so a unit compiled under the other flush mode earlier on
    # this host could come back instead.
    from gpuwm.nvrtc_cache_key import compile_program
    blob = compile_program(source, options, target="cubin")
    # CuPy's binding drops the object's final byte (a NUL); nvdisasm wants it.
    return blob + b"\x00"


def kernel_digests(blob: bytes, nvdisasm: str) -> dict[str, str]:
    with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as fh:
        fh.write(blob)
        path = fh.name
    try:
        text = subprocess.run([nvdisasm, "-c", path], capture_output=True,
                              text=True, check=True).stdout
    finally:
        Path(path).unlink()
    out: dict[str, list[str]] = {}
    current = None
    for line in text.splitlines():
        if line.startswith(".text."):
            current = line[len(".text."):].rstrip(":")
            out.setdefault(current, [])
            continue
        m = _INSN.match(line)
        if m and current is not None:
            insn = re.sub(r"__internal_\d+_", "__internal_", m.group(1))
            insn = re.sub(r"\.L_x_\d+", ".L_x", insn)
            insn = re.sub(r"\s+", " ", insn)
            out[current].append(insn)
    return {k: {"sequence": _sha(v), "multiset": _sha(sorted(map(_unreg, v)))}
            for k, v in out.items()}


def _sha(lines) -> str:
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def _unreg(insn: str) -> str:
    """The instruction with register names and branch targets erased.

    ``x / C`` hands ptxas the divisor as an immediate and ``__fdiv_rn`` a
    register holding the same immediate, so the allocator can number and
    order the same instructions differently.  With registers erased every
    opcode, modifier and immediate is still compared."""
    insn = re.sub(r"\b(?:U?R|U?P|B)\d+\b", "r", insn)
    insn = re.sub(r"\.reuse\b", "", insn)
    return re.sub(r"`\([^)]*\)", "`()", insn)


def spelled_as_slash(source: str) -> str:
    """Every ``__fdiv_rn(a, b)`` call rewritten as ``((a) / (b))``.

    Applied to both trees, PTX identity for a target that keeps ``/`` as
    ``div.rn`` then says the two trees differ only in which of the two
    spellings of the same division each site uses: no operand was regrouped,
    retyped or dropped by the rewrite."""
    out, i, key = [], 0, "__fdiv_rn("
    while True:
        j = source.find(key, i)
        if j < 0:
            out.append(source[i:])
            return "".join(out)
        out.append(source[i:j])
        k, depth, comma = j + len(key), 1, None
        while depth:
            c = source[k]
            if c in "([{":
                depth += 1
            elif c in ")]}":
                depth -= 1
            elif c == "," and depth == 1:
                comma = k
            k += 1
        a = spelled_as_slash(source[j + len(key):comma])
        b = spelled_as_slash(source[comma + 1:k - 1])
        out.append(f"(({a}) / ({b}))")
        i = k


_VREG = re.compile(r"%[a-z]+\d+")


def renumbered(ptx: str) -> str:
    """Virtual registers renamed in order of first use, per function.

    A compound ``x /= c`` spelled ``x = x / c`` is the same operation but can
    leave NVVM numbering its virtual registers one apart."""
    out, names = [], {}
    for line in ptx.splitlines():
        if line.startswith((".visible .entry", ".entry", ".func",
                            ".visible .func", ".weak .func")):
            names = {}
        out.append(_VREG.sub(
            lambda m: names.setdefault(m.group(0), f"%v{len(names)}"), line))
    return "\n".join(out)


#: Every ``__fdiv_rn`` routed through an overload set that refuses any
#: operand that is not already ``float`` (an ``int`` or ``double`` would
#: silently convert).  Prepended by :func:`type_checked`.
_FLOAT_ONLY = (
    "__device__ __forceinline__ float a146_fdiv(float a, float b)"
    " { return __fdiv_rn(a, b); }\n"
    "template <class A, class B> __device__ float a146_fdiv(A, B) = delete;\n")


def type_checked(source: str) -> str:
    return _FLOAT_ONLY + source.replace("__fdiv_rn(", "a146_fdiv(").replace(
        "{ return a146_fdiv(a, b); }", "{ return __fdiv_rn(a, b); }", 1)


def ptx_digests(arch: str, root=ROOT) -> dict[str, str]:
    """compute_* PTX digest of every unit spelled with ``/`` throughout."""
    from tools.literal_division_census import compile_ptx
    out = {}
    for unit in production_units(root):
        opts = tuple(unit.options) + ("-ftz=true", f"-arch={arch}")
        try:
            ptx = compile_ptx(spelled_as_slash(unit.source), opts)
        except Exception as exc:
            out[unit.key] = "failed: " + str(exc).splitlines()[0][:120]
            continue
        out[unit.key] = hashlib.sha256(renumbered(ptx).encode()).hexdigest()
    return out


def type_check(arch: str, root=ROOT) -> dict[str, str]:
    from tools.literal_division_census import compile_ptx
    out = {}
    for unit in production_units(root):
        opts = tuple(unit.options) + ("-ftz=true", f"-arch={arch}")
        try:
            compile_ptx(type_checked(unit.source), opts)
            out[unit.key] = "float operands only"
        except Exception as exc:
            from cupy.cuda import nvrtc  # noqa: F401
            out[unit.key] = "refused: " + str(exc).splitlines()[0][:160]
    return out


_DIV_OR_CVT = re.compile(r"^\s*((?:div|rem|cvt)\.[a-z0-9.]+)\s", re.M)


def division_histograms(arch: str, root=ROOT) -> dict[str, dict[str, int]]:
    """Count of every div/rem/cvt opcode per unit, in the real sources.

    The slash-spelled identity cannot see an integer or double operand that
    a rewrite handed to ``__fdiv_rn`` (it would spell the same ``/``); the
    real PTX can: such an operand adds a ``cvt`` and trades an integer or
    f64 ``div`` for an f32 one, so equal histograms rule it out."""
    from collections import Counter

    from tools.literal_division_census import compile_ptx
    out = {}
    for unit in production_units(root):
        opts = tuple(unit.options) + ("-ftz=true", f"-arch={arch}")
        try:
            ptx = compile_ptx(unit.source, opts)
        except Exception as exc:
            out[unit.key] = {"failed": 1}
            continue
        out[unit.key] = dict(sorted(Counter(_DIV_OR_CVT.findall(ptx)).items()))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="sm_89")
    ap.add_argument("--nvdisasm")
    ap.add_argument("--json", required=True)
    ap.add_argument("--root", default=str(ROOT),
                    help="repository root of the tree to compile")
    ap.add_argument("--slash-ptx", action="store_true",
                    help="digest compute_* PTX of the '/'-spelled sources")
    ap.add_argument("--div-histogram", action="store_true",
                    help="div/rem/cvt opcode counts of the real sources")
    ap.add_argument("--type-check", action="store_true",
                    help="refuse any __fdiv_rn operand that is not float")
    args = ap.parse_args(argv)
    if args.type_check:
        res = type_check(args.arch.replace("sm_", "compute_"), args.root)
        Path(args.json).write_text(json.dumps(res, indent=1))
        print(sum(v.startswith("refused") for v in res.values()), "refused")
        return 0
    if args.div_histogram:
        hist = division_histograms(args.arch.replace("sm_", "compute_"),
                                   args.root)
        Path(args.json).write_text(json.dumps(hist, indent=1))
        print(f"{len(hist)} units")
        return 0
    if args.slash_ptx:
        digests = ptx_digests(args.arch.replace("sm_", "compute_"),
                              args.root)
        Path(args.json).write_text(json.dumps(digests, indent=1))
        print(f"{len(digests)} units")
        return 0
    result, failed = {}, {}
    for unit in production_units(args.root):
        opts = tuple(unit.options) + ("-ftz=true", f"-arch={args.arch}")
        try:
            result[unit.key] = kernel_digests(cubin(unit.source, opts),
                                              args.nvdisasm)
        except Exception as exc:
            failed[unit.key] = str(exc).splitlines()[0][:200]
    Path(args.json).write_text(json.dumps({"arch": args.arch, "units": result,
                                           "failed": failed}, indent=1))
    print(f"{len(result)} units, {sum(len(v) for v in result.values())} "
          f"kernels, {len(failed)} failed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
