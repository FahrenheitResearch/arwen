"""A146: every float division by a compile-time constant the compiler sees.

NVRTC's NVVM, targeting compute_100 and compute_120 (Blackwell) under
``-ftz=true`` (which CuPy appends to every ``RawModule``), replaces a float
division by a compile-time constant with a multiplication by the rounded
reciprocal: ``x / 3.0f`` becomes ``mul.ftz.f32 x, 0f3EAAAAAB``, which is not
IEEE division (a third of all ``x / 3.0f`` quotients land one ULP off).
Every NVRTC measured (12.9, 13.0, 13.3, 13.4) does it; compute_90 and older
keep ``div.rn``.  A source grep cannot find every such site: a divisor can
reach the division through a macro, a ``const``, an integer literal, a
constant-folded product, or a helper inlined with a literal argument.  The
compiler can.  This census compiles each production translation unit for
compute_89, where the same division survives as ``div.rn[.ftz].f32`` with an
immediate divisor, and lists those instructions with the source line
``-lineinfo`` attaches to them.  A power-of-two, zero or infinite divisor
has an exact reciprocal, so either spelling gives the IEEE quotient, and it
is not listed.

The chosen form, ``__fdiv_rn(x, C)``, reaches PTX as a ``div.rn`` whose
divisor is a register, so it is never listed.

A compute_89 immediate is not the whole set compute_120 rewrites.  A divisor
compute_89 keeps in a register can still be a constant on compute_120: NVRTC
13.4.92 constant-folds ``tgammaf(4.0f)`` there, which compute_89 calls at run
time, and a pair of constant divisions in the two arms of a branch can reach
compute_89 as one ``div.rn`` of a selected register.  :func:`rewrite_sites`
compiles the unit for compute_120 twice, under ``-ftz=false`` (where a float
division by a constant stays a ``div.rn``) and under ``-ftz=true``, and lists
every source line whose IEEE quotients (``div.rn`` and ``rcp.rn``) fall under
``-ftz=true`` while its multiplies (or fused multiply-adds) gain an immediate
with an inexact reciprocal there: the division became a reciprocal multiply.
Constant operands are counted, including one-write ``mov.f32`` registers
within each PTX function. An existing coefficient of the same value must
not hide another occurrence introduced by a reciprocal rewrite.

The reference is the same architecture (A174).  It used to be compute_89,
and NVRTC 12.9.86, the default ``gpuwm[gpu]`` compiler, compiles ``powf`` and
``logf`` differently for the two architectures (as separate functions on
compute_120, inlined on compute_89), so lines of noah, nssl2_nucond, shinhong
and thompson that hold no constant division lost ``div.rn`` and gained
immediates between the two and were listed.  Over every production unit under
both compilers, the flush-mode comparison listed one line that is not a
rewrite, a reciprocal: NVRTC 13.4.92 compiles thompson.cu's
``1.0f / (1004.0f * (1.0f + 0.887f * qv0))`` as a ``div.rn`` under
``-ftz=false`` and as an ``rcp.rn`` under ``-ftz=true``, its sign folded into
a new multiply by -1004.  ``rcp.rn`` is the IEEE reciprocal, so it is counted
as the quotient it is, and that line is not listed.  The flush modes can
still inline differently (13.4.92 compiles two of p3_device's functions out
of line under ``-ftz=true`` only; no line of it is listed), so a new listing
is read from the PTX before it is taken for a rewrite.

Usage (needs CuPy's NVRTC; no GPU is touched when CUDA is hidden)::

    CUDA_VISIBLE_DEVICES= python -m tools.literal_division_census [--json out]
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KDIR = ROOT / "gpuwm" / "core" / "kernels"

#: Units compiled with ``--ftz=false`` through ``compile_using_nvrtc``: the
#: rewrite needs ``-ftz=true``, so their constant divisions stay IEEE.
FTZ_FALSE_UNITS = ("rrtmg_legacy", "rrtmg_legacy_device", "rrtmg_lw",
                   "rrtmg_mcica", "rrtmg_sw")

_DIV_IMM = re.compile(
    r"^\s*div\.rn(?:\.ftz)?\.f32\s+%\w+,\s*%\w+,\s*0[fF]([0-9A-Fa-f]{8});")
_DIV_ANY = re.compile(r"^\s*div\.rn(?:\.ftz)?\.f32\s")
#: An IEEE-rounded float quotient: a ``div.rn``, or the ``rcp.rn`` a
#: reciprocal of a runtime value compiles to under one flush mode and not
#: the other (A174).
_QUOTIENT = re.compile(r"^\s*(?:div|rcp)\.rn(?:\.ftz)?\.f32\s")
_MUL_OP = re.compile(r"^\s*(?:mul|fma)(?:\.rn)?(?:\.ftz)?\.f32\s")
_IMMEDIATE = re.compile(r"(?<![\w.])0[fF]([0-9A-Fa-f]{8})\b")
_LOC = re.compile(r"^\s*\.loc\s+(\d+)\s+(\d+)\s+(\d+)(.*)$")
_INLINED_AT = re.compile(r"inlined_at\s+(\d+)\s+(\d+)\s+(\d+)")

_FUNCTION_START = re.compile(
    r"(?=^[ \t]*(?:(?:\.visible|\.weak|\.extern)\s+)?\.(?:entry|func)\b)",
    re.MULTILINE)
_INSTRUCTION = re.compile(
    r"^\s*(?:@\S+\s+)?([A-Za-z][\w.]*)\s+([^;]+);")
_DESTINATION = re.compile(r"^\s*([({][^)}]*[)}]|[^,]+)")
_REGISTER = re.compile(r"%\w+")


@dataclass
class Unit:
    key: str
    source: str
    options: tuple[str, ...]
    #: (first assembled line, file label, first line in that file)
    segments: list[tuple[int, str, int]] = field(default_factory=list)

    def locate(self, line: int) -> tuple[str, int]:
        label, base, first = "<assembled>", 1, 1
        for start, lab, file_first in self.segments:
            if line >= start:
                label, base, first = lab, start, file_first
        return label, line - base + first


def _lines(text: str) -> int:
    return text.count("\n")


def _segmented(key, pieces, options):
    """Assemble ``pieces`` [(label, text)] and remember where each starts."""
    segs, line = [], 1
    for label, text in pieces:
        segs.append((line, label, 1))
        line += _lines(text)
    return Unit(key, "".join(text for _label, text in pieces), options, segs)


def _preamble_pieces():
    from gpuwm.core.constants import CUDA_DEFINES
    defines = "".join(f"#define {k} {float(v)!r}f\n"
                      for k, v in CUDA_DEFINES.items())
    common = (KDIR / "common.cuh").read_text(encoding="utf-8")
    return [("<CUDA_DEFINES>", defines), ("gpuwm/core/kernels/common.cuh",
                                          common + "\n")]


def production_units() -> list[Unit]:
    """Every -ftz=true translation unit the package compiles, as compiled."""
    from gpuwm.core import kernels as K
    from gpuwm.core import noahmp_kernel_sources as N

    units: list[Unit] = []
    pre = _preamble_pieces()
    assert "".join(t for _l, t in pre) == K._preamble(), \
        "census preamble drifted from the loader's"

    def kfile(name):
        return (f"gpuwm/core/kernels/{name}",
                (KDIR / name).read_text(encoding="utf-8"))

    noahmp_parts = set()
    for parts in N.NOAHMP_TRANSLATION_UNITS.values():
        noahmp_parts.update(parts)
    for path in sorted(KDIR.glob("*.cu")):
        name = path.stem
        if name in noahmp_parts:
            continue
        extra = [kfile(h) for h in K.EXTRA_HEADERS.get(name, ())]
        unit = _segmented(f"kernels:{name}", pre + extra + [kfile(path.name)],
                          ("-std=c++17",))
        if unit.source != K.module_source(name):
            raise AssertionError(f"census source for {name} drifted")
        units.append(unit)
    # gf above 40 levels compiles with an integer tier; 60 is one such tier.
    extra = [kfile(h) for h in K.EXTRA_HEADERS.get("gf", ())]
    gf = _segmented("kernels:gf[GF_KMAX=60]",
                    pre + extra + [("<GF_KMAX>", "#define GF_KMAX 60\n"),
                                   kfile("gf.cu")], ("-std=c++17",))
    assert gf.source == K.module_source_int_defines("gf", (("GF_KMAX", 60),))
    units.append(gf)
    for name, parts in N.NOAHMP_TRANSLATION_UNITS.items():
        ru = N.runtime_unit(name)
        pieces = ([] if ru.preamble_sha256 != N._sha(K._preamble()) else pre)
        pieces = pieces + [kfile(p + ".cu") for p in parts]
        unit = _segmented(f"noahmp:{name}", pieces, tuple(ru.options))
        if unit.source != ru.source:
            # Composition differs from the simple concatenation (a unit with
            # its own prefix); keep the real source and locate lines by
            # searching rather than guessing.
            unit = Unit(f"noahmp:{name}", ru.source, tuple(ru.options), [])
        units.append(unit)
    from gpuwm.core import ruc_tier as RT
    for nzs in (6, 9):
        units.append(Unit(f"ruc_tier[nzs={nzs}]", RT.ruc_fused_source(nzs),
                          ("-std=c++17",), []))
    from gpuwm.core import p3_device as P3
    p3 = _segmented("p3_device", pre + [
        (str(P3._LIBM_SOURCE.relative_to(ROOT).as_posix()),
         P3._LIBM_SOURCE.read_text(encoding="utf-8")),
        (str(P3._P3_SOURCE.relative_to(ROOT).as_posix()),
         P3._P3_SOURCE.read_text(encoding="utf-8"))],
        tuple(P3.DEFAULT_OPTIONS))
    assert p3.source == P3.p3_source()
    units.append(p3)
    return units


def compile_ptx(source: str, options: tuple[str, ...]) -> str:
    # Keyed by its options (A160): NVRTC's cache ignores -ftz for a fixed
    # program name, so one planted source compiled under both flush modes
    # on a host with a GPU could read the other mode's PTX.
    from gpuwm.nvrtc_cache_key import compile_program
    ptx = compile_program(source, options, target="ptx")
    return ptx.decode() if isinstance(ptx, bytes) else ptx


def _is_exact_divisor(bits: int) -> bool:
    """A power of two, zero or infinity: ``x * (1/C)`` equals ``x / C``.

    Division by a power of two only moves the exponent (the reciprocal is
    exact), and a zero or infinite divisor has an exact reciprocal too, so
    the rewrite gives the IEEE quotient for these (a zero divisor with a zero
    numerator is NaN either way)."""
    return (bits & 0x007FFFFF) == 0


def census_unit(unit: Unit, arch: str = "compute_89") -> dict:
    """Constant-divisor float divisions of one unit, with source lines."""
    opts = tuple(unit.options) + ("-ftz=true", "-lineinfo", f"-arch={arch}")
    ptx = compile_ptx(unit.source, opts)
    sites, loc, total = [], None, 0
    for text in ptx.splitlines():
        m = _LOC.match(text)
        if m:
            loc = (int(m.group(2)), int(m.group(3)),
                   _INLINED_AT.findall(m.group(4)))
            continue
        if _DIV_ANY.match(text):
            total += 1
        m = _DIV_IMM.match(text)
        if not m:
            continue
        bits = int(m.group(1), 16)
        if _is_exact_divisor(bits):
            continue
        divisor = struct.unpack("<f", struct.pack("<I", bits))[0]
        line, column, chain = loc if loc else (0, 0, [])
        label, file_line = unit.locate(line)
        outer = None
        if chain:
            _f, oline, _c = chain[-1]
            outer = unit.locate(int(oline))
        sites.append({"file": label, "line": file_line, "column": column,
                      "assembled_line": line, "divisor": repr(divisor),
                      "inlined_into": list(outer) if outer else None})
    return {"unit": unit.key, "arch": arch, "div_rn_f32": total,
            "constant_divisor_sites": sites}


def _constant_registers(ptx: str) -> dict[str, int]:
    """Registers written once, by ``mov.f32`` of a float immediate.

    The caller passes one PTX function, since register names are local to
    it.  Reject every multiply-written register, including conditional
    assignments, rather than guessing a constant across control flow.
    """
    writes: dict[str, list[int | None]] = {}
    for text in ptx.splitlines():
        instruction = _INSTRUCTION.match(text)
        if not instruction:
            continue
        opcode, operands = instruction.groups()
        destination = _DESTINATION.match(operands)
        if not destination:
            continue
        registers = _REGISTER.findall(destination.group(1))
        value = None
        if opcode == "mov.f32" and len(registers) == 1:
            immediate = _IMMEDIATE.fullmatch(
                operands[destination.end():].lstrip(" ,"))
            if immediate:
                value = int(immediate.group(1), 16)
        for register in registers:
            writes.setdefault(register, []).append(value)
    return {register: values[0] for register, values in writes.items()
            if len(values) == 1 and values[0] is not None}


def _line_counts(ptx: str) -> dict[tuple[int, int], tuple[int, Counter[int]]]:
    """(file, line) -> (IEEE quotient count, constant float operands of mul
    and fma.f32), by the innermost source line ``-lineinfo`` gives each
    instruction.  A one-write ``mov.f32`` register and a direct immediate
    are the same operand, as NVRTC's targets can use either spelling.
    Count occurrences so a new reciprocal cannot hide behind an existing
    coefficient with the same value."""
    counts: dict[tuple[int, int], list] = {}
    for function in _FUNCTION_START.split(ptx):
        constants = _constant_registers(function)
        loc = None
        for text in function.splitlines():
            m = _LOC.match(text)
            if m:
                loc = (int(m.group(1)), int(m.group(2)))
                continue
            if loc is None:
                continue
            if _QUOTIENT.match(text):
                counts.setdefault(loc, [0, Counter()])[0] += 1
            elif _MUL_OP.match(text):
                operands = text.split(",", 1)[1]
                values = [int(bits, 16)
                          for bits in _IMMEDIATE.findall(operands)]
                values.extend(constants[register]
                              for register in _REGISTER.findall(operands)
                              if register in constants)
                counts.setdefault(loc, [0, Counter()])[1].update(values)
    return {key: (n, imms) for key, (n, imms) in counts.items()}


def _plain_division_on(line_text: str) -> bool:
    """Does this source line hold a ``/`` outside comments and strings?  A
    line whose only divisions are ``__fdiv_rn`` calls (or helpers built on
    it) cannot be rewritten; compute_120 folding such a call on constants
    gives the IEEE quotient."""
    code = line_text
    for pattern in (r"//.*$", r"/\*.*?\*/", r'"(?:\\.|[^"\\])*"'):
        code = re.sub(pattern, " ", code)
    return "/" in code


def rewrite_sites(unit: Unit, arch: str = "compute_120") -> dict:
    """Source lines whose float division ``arch`` compiles under
    ``-ftz=true`` as a multiply by an immediate where the same architecture
    under ``-ftz=false`` keeps a ``div.rn``: the line loses an IEEE quotient
    (``div.rn`` or ``rcp.rn``) under ``-ftz=true`` and its multiplies gain an
    immediate the reference does not use there, one whose reciprocal is not
    exact (a power of two, zero or infinity rewrite to the IEEE quotient).
    A line that loses the division by folding a constant ``__fdiv_rn`` is
    not listed.

    The reference is the same architecture so that only the flush mode
    differs between the two builds (A174): a compute_89 reference read
    NVRTC 12.9.86's compute_120 code for ``powf`` and ``logf``, which is not
    its compute_89 code, as rewritten divisions."""
    base = tuple(unit.options) + ("-lineinfo", f"-arch={arch}")
    before = _line_counts(compile_ptx(unit.source, base + ("-ftz=false",)))
    after = _line_counts(compile_ptx(unit.source, base + ("-ftz=true",)))
    source_lines = unit.source.splitlines()
    sites = []
    for key in sorted(set(before) | set(after)):
        div_old, imm_old = before.get(key, (0, Counter()))
        div_new, imm_new = after.get(key, (0, Counter()))
        gained = sorted(bits for bits in imm_new - imm_old
                        if not _is_exact_divisor(bits))
        if div_new >= div_old or not gained:
            continue
        if not 0 < key[1] <= len(source_lines):
            continue    # no source line to attribute a plain division to
        text = source_lines[key[1] - 1]
        if not _plain_division_on(text):
            continue
        label, file_line = unit.locate(key[1])
        sites.append({"file": label, "line": file_line,
                      "assembled_line": key[1],
                      "quotients": [div_old, div_new],
                      "gained_immediates": [
                          repr(struct.unpack("<f", struct.pack("<I", b))[0])
                          for b in gained]})
    return {"unit": unit.key, "arch": arch,
            "flush_modes": ["-ftz=false", "-ftz=true"],
            "rewritten_sites": sites}


def census(units=None) -> list[dict]:
    return [census_unit(u) for u in (units or production_units())]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", help="write the full census here")
    ap.add_argument("--arch", default=None,
                    help="compute_89 for the immediate census, compute_120 "
                         "for --rewrites (the defaults)")
    ap.add_argument("--rewrites", action="store_true",
                    help="compare -ftz=false with -ftz=true line by line "
                         "on --arch instead")
    args = ap.parse_args(argv)
    rows, failed = [], []
    key = "rewritten_sites" if args.rewrites else "constant_divisor_sites"
    arch = args.arch or ("compute_120" if args.rewrites else "compute_89")
    for unit in production_units():
        try:
            rows.append(rewrite_sites(unit, arch) if args.rewrites
                        else census_unit(unit, arch))
        except Exception as exc:  # a unit NVRTC refuses is reported, not hidden
            failed.append((unit.key, str(exc).splitlines()[0][:200]))
    for unit_key, why in failed:
        print(f"COMPILE FAILED {unit_key}: {why}")
    total = 0
    for row in rows:
        n = len(row[key])
        total += n
        if n:
            print(f"{row['unit']}: {n}")
            for site in row[key]:
                print(f"    {site['file']}:{site['line']}")
    print(f"total {key.replace('_', ' ')}: {total}")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1) + "\n",
                                   encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
