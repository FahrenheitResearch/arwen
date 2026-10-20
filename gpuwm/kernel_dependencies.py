"""Bind literal external includes; refuse dependencies requiring preprocessing.

This performs no compiler or device work. Builtin NVRTC headers are bound by
loaded compiler bytes. All matching external header candidates are hashed,
including nested literal includes, so search-order shadowing changes the key.
Macro includes and forced include options keep the ordinary compiler route.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re

_BUILTIN = frozenset({"assert.h", "ctype.h", "float.h", "limits.h", "math.h",
    "stddef.h", "stdint.h", "stdbool.h", "stdio.h", "stdlib.h", "string.h",
    "cmath", "cstddef", "cstdint", "cfloat", "climits", "cstring"})


def forced_include(options):
    return any(str(option).startswith(("--pre-include", "-include")) for option in options)


def header_dependencies(source, options=(), filename="kern.cu"):
    """Return actual header digests, or raise when their identity is unbound."""
    if forced_include(options):
        raise ValueError("forced include options require the ordinary compiler")
    roots = [Path.cwd()]
    path = Path(str(filename))
    if path.is_absolute():
        roots.insert(0, path.parent)
    options = list(map(str, options))
    for i, option in enumerate(options):
        if option in ("-I", "--include-path") and i + 1 < len(options):
            roots.append(Path(options[i + 1]))
        elif option.startswith("-I") and len(option) > 2:
            roots.append(Path(option[2:]))
        elif option.startswith("--include-path="):
            roots.append(Path(option.split("=", 1)[1]))
    found = {}
    def visit(text, local):
        text = re.sub(r"\\\r?\n", "", text)
        if 'R"' in text or '??' in text:
            raise ValueError("nonstandard directive spelling requires the ordinary compiler")
        tokens = re.compile(r"\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|/\*.*?\*/|//[^\n]*", re.S)
        text = tokens.sub(lambda match: " " + "\n" * match.group().count("\n")
                          if match.group().startswith(("/*", "//")) else match.group(), text)
        text = text.replace("%:", "#")
        if "__has_include" in text:
            raise ValueError("conditional header discovery requires the ordinary compiler")
        for directive, operand in re.findall(r"^[^\S\n]*#[^\S\n]*(include_next|include|import)\b([^\n]*)", text, re.M):
            match = re.fullmatch(r'\s*(?:"([^"\n]+)"|<([^>\n]+)>)\s*', operand)
            if directive != "include" or match is None:
                raise ValueError("macro or nonliteral include requires the ordinary compiler")
            name = match.group(1) or match.group(2)
            candidates = []
            for directory in [local, *roots]:
                candidate = directory / name
                if candidate.is_file():
                    resolved = candidate.resolve()
                    if resolved not in candidates:
                        candidates.append(resolved)
            if not candidates:
                if match.group(2) and name in _BUILTIN:
                    continue
                raise ValueError(f"external header cannot be bound: {name}")
            for candidate in candidates:
                key = str(candidate)
                if key in found:
                    continue
                if len(found) >= 1024:
                    raise ValueError("header dependency limit exceeded")
                data = candidate.read_bytes()
                found[key] = hashlib.sha256(data).hexdigest()
                visit(data.decode("utf-8", "replace"), candidate.parent)
    visit(source, roots[0])
    return [{"path": path, "sha256": digest} for path, digest in sorted(found.items())]
