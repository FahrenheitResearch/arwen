"""Floating expression fingerprints for PTX contraction diagnostics.

This traces explicit register definitions rather than counting FMA opcodes.
The two expressions fma(a,b,round(c*d)) and fma(c,d,round(a*b)) have the
same opcode counts but distinct fingerprints. Parameter names, constant bits
and load addresses anchor leaves; register numbers do not. The analysis is a
local compiler diagnostic. It does not solve control-flow phi nodes or prove
memory aliasing, so a changed fingerprint still requires source review/replay.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import re

_ENTRY = re.compile(r"\.(?:entry|func)\s+(?:\([^)]*\)\s*)?([\w$]+)\s*\(")
_LOC = re.compile(r"^\s*\.loc\s+(\d+)\s+(\d+)\s+(\d+)(.*)$")
_INLINE = re.compile(r"inlined_at\s+(\d+)\s+(\d+)\s+(\d+)")
_INSTRUCTION = re.compile(r"^\s*(?:@\S+\s+)?([A-Za-z][\w.]*)\s+([^;]+);")
_REGISTER = re.compile(r"%[\w]+")
_FLOAT = re.compile(r"^(?:fma|mad|mul|add|sub|div|rcp|sqrt)(?:\.[\w]+)*\.f(?:32|64)$")


def _digest(value) -> str:
    return hashlib.sha256(repr(value).encode()).hexdigest()[:24]


def _rounding_op(opcode: str) -> str:
    pieces = opcode.split(".")
    if pieces[0] in ("fma", "mad", "mul", "add", "sub", "div", "sqrt"):
        if not any(rounding in pieces for rounding in ("rn", "rz", "rp", "rm", "approx")):
            pieces.insert(1, "rn")
    return ".".join(pieces)


def ptx_dataflow_signatures(ptx: str, statement_lines=None) -> dict[tuple, Counter]:
    """FMA expression DAG fingerprints, grouped by source/inlining site.

    Every DAG is interned as a fixed-size hash, so repeated definitions cannot
    expand an expression tree exponentially. A destination register overwrite
    updates its definition. Definitions at branch joins are not an SSA proof;
    this restriction is stated in the report rather than hidden by a count.
    """
    signatures, definitions = {}, {}
    function, location = "<global>", (0, 0, 0, ())
    statement_lines = statement_lines or {}

    def operand(text):
        text = text.strip()
        if text in definitions:
            return definitions[text]
        if text.startswith("-") and text[1:] in definitions:
            return _digest(("neg", definitions[text[1:]]))
        if _REGISTER.fullmatch(text):
            # An unbound register cannot anchor operand identity. The unknown
            # type is stable across register renaming, and visibly weaker.
            return _digest(("unresolved", re.sub(r"\d+$", "", text)))
        resolved = _REGISTER.sub(lambda match: definitions.get(
            match[0], "<unresolved>"), text)
        return _digest(("literal_or_address", resolved.upper()))

    for text in ptx.splitlines():
        match = _ENTRY.search(text)
        if match:
            function = match[1]
            definitions = {}
            location = (0, 0, 0, ())
        match = _LOC.match(text)
        if match:
            file, line, column = (int(match[i]) for i in (1, 2, 3))
            chain = tuple(tuple(int(x) for x in item)
                          for item in _INLINE.findall(match[4]))
            if statement_lines:
                if file == 1:
                    line, column = statement_lines.get(line, line), 0
                chain = tuple((file, statement_lines.get(line, line), 0)
                              if file == 1 else (file, line, column)
                              for file, line, column in chain)
            location = (file, line, column, chain)
            continue
        match = _INSTRUCTION.match(text)
        if not match:
            continue
        opcode, arguments = match[1], match[2]
        parts = [part.strip() for part in arguments.split(",")]
        if not parts or not _REGISTER.fullmatch(parts[0]):
            continue
        destination, arguments = parts[0], parts[1:]
        if opcode.startswith("st."):
            continue
        values = tuple(operand(argument) for argument in arguments)
        if opcode.startswith(("mov.", "cvta.")) and len(values) == 1:
            definition = values[0]
        elif opcode.startswith("ld."):
            # PTX can spell an identical 32-bit load b32 on one target and
            # f32 on another. Cache qualifiers do not change its words.
            width = opcode.split(".")[-1].lstrip("bsuf")
            space = opcode.split(".")[1]
            definition = _digest(("load", space, width, values))
        else:
            normalized = _rounding_op(opcode) if _FLOAT.match(opcode) else opcode
            if opcode.startswith(("mul.", "add.")) and len(values) == 2:
                values = tuple(sorted(values))
            elif opcode.startswith(("fma.", "mad.")) and len(values) == 3:
                # Only the two exact-product operands commute. Moving a
                # rounded product between product and addend changes the DAG.
                values = tuple(sorted(values[:2])) + values[2:]
            definition = _digest((normalized, values))
        definitions[destination] = definition
        if opcode.startswith(("fma.", "mad.")) and _FLOAT.match(opcode):
            site = (function, *location)
            signatures.setdefault(site, Counter())[definition] += 1
    return signatures


def dataflow_differences(left: dict[tuple, Counter], right: dict[tuple, Counter]) -> list[dict]:
    differences = []
    for site in sorted(set(left) | set(right)):
        before, after = left.get(site, Counter()), right.get(site, Counter())
        if before != after:
            function, file, line, column, chain = site
            differences.append({"function": function, "file_number": file,
                                "assembled_line": line, "column": column,
                                "inline_chain": [list(item) for item in chain],
                                "left_dags": dict(sorted(before.items())),
                                "right_dags": dict(sorted(after.items()))})
    return differences
