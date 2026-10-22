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


def include_roots(options):
    roots = []
    options = list(map(str, options))
    for i, option in enumerate(options):
        if option in ("-I", "--include-path") and i + 1 < len(options):
            roots.append(Path(options[i + 1]))
        elif option.startswith("-I") and len(option) > 2:
            roots.append(Path(option[2:]))
        elif option.startswith("--include-path="):
            roots.append(Path(option.split("=", 1)[1]))
    return roots


def include_tree_dependencies(options, snapshots):
    """Conservative binding for header macros, including CCCL dispatch.

    NVRTC has no host system-header search. Bind all files in every explicit
    include root, in search order, including extensionless C++ headers. Roots
    are snapshotted once per compiler session, just like its loaded libraries;
    a new process rehashes bytes and detects changed or added headers.
    """
    from concurrent.futures import ThreadPoolExecutor
    result = []
    roots = include_roots(options)
    resolved_roots = [root.resolve() for root in roots]
    for root in roots:
        key = str(root.resolve())
        if key not in snapshots:
            paths = sorted(p for p in root.rglob("*") if p.is_file())
            def read(path):
                data = path.read_bytes()
                # A computed dependency outside these roots cannot safely be
                # served from a tree snapshot. Ordinary NVRTC still handles it.
                directives = re.findall(rb'^\s*#\s*(?:include|define)\b[^\n]*', data, re.M)
                for directive in directives:
                    for name in re.findall(rb'["<]((?:/|\.\./)[^">\n]*)[">]', directive):
                        target = (path.parent / name.decode()).resolve()
                        if not any(target.is_relative_to(base) for base in resolved_roots):
                            raise ValueError(f"header may escape include roots: {path.name}")
                return (path.relative_to(root).as_posix(), hashlib.sha256(data).hexdigest())
            with ThreadPoolExecutor(min(16, max(1, len(paths)))) as pool:
                rows = list(pool.map(read, paths))
            from gpuwm.compiler_metadata import _digest
            snapshots[key] = _digest(rows)
        result.append(dict(tree_sha256=snapshots[key]))
    if not result:
        raise ValueError("computed header dependency has no explicit include roots")
    return result

_BUILTIN = frozenset({"assert.h", "ctype.h", "float.h", "limits.h", "math.h",
    "stddef.h", "stdint.h", "stdbool.h", "stdio.h", "stdlib.h", "string.h",
    "cmath", "cstddef", "cstdint", "cfloat", "climits", "cstring"})


def forced_include(options):
    return any(str(option).startswith(("--pre-include", "-include")) for option in options)


def header_dependencies(source, options=(), filename="kern.cu", *, portable=False):
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
    missing = set()
    texts = []
    aliases = {"__has_include"}
    discoveries = set()
    def resolve(name, local, builtin=False):
        candidates = []
        for directory in [local, *roots]:
            candidate = directory / name
            if candidate.is_file():
                resolved = candidate.resolve()
                if resolved not in candidates:
                    candidates.append(resolved)
        if not candidates:
            if builtin and name in _BUILTIN:
                return
            if not portable:
                raise ValueError(f"external header cannot be bound: {name}")
            # Includes inside false guards (including already inlined headers)
            # are not dependencies. Bind absence too: installing the header
            # must invalidate a previous __has_include decision.
            missing.add(name)
        for candidate in candidates:
            key = str(candidate)
            if key in found:
                continue
            if len(found) >= 4096:
                raise ValueError("header dependency limit exceeded")
            data = candidate.read_bytes()
            found[key] = hashlib.sha256(data).hexdigest()
            visit(data.decode("utf-8", "replace"), candidate.parent)

    def visit(text, local):
        text = re.sub(r"\\\r?\n", "", text)
        if 'R"' in text or '??' in text:
            raise ValueError("nonstandard directive spelling requires the ordinary compiler")
        tokens = re.compile(r"\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|/\*.*?\*/|//[^\n]*", re.S)
        text = tokens.sub(lambda match: " " + "\n" * match.group().count("\n")
                          if match.group().startswith(("/*", "//")) else match.group(), text)
        text = text.replace("%:", "#")
        if "__has_include" in text and not portable:
            raise ValueError("conditional header discovery requires the ordinary compiler")
        texts.append((text, local))
        for directive, operand in re.findall(r"^[^\S\n]*#[^\S\n]*(include_next|include|import)\b([^\n]*)", text, re.M):
            match = re.fullmatch(r'\s*(?:"([^"\n]+)"|<([^>\n]+)>)\s*', operand)
            if directive != "include" or match is None:
                raise ValueError(f"macro or nonliteral include requires the ordinary compiler: {directive} {operand}")
            name = match.group(1) or match.group(2)
            resolve(name, local, bool(match.group(2)))
    visit(source, roots[0])
    if portable:
        # CCCL wraps __has_include in a single-argument identity macro. Bind
        # every literal discovery, even in an inactive branch; refuse computed
        # names rather than guessing a preprocessor result.
        while True:
            before = (len(texts), len(aliases), len(discoveries))
            for text, local in list(texts):
                for alias, arg, target, value in re.findall(
                        r'^\s*#\s*define\s+(\w+)\((\w+)\)\s+(\w+)\((\w+)\)\s*$', text, re.M):
                    if target in aliases and arg == value:
                        aliases.add(alias)
                for line in text.splitlines():
                    if re.match(r'\s*#\s*define\s+(?:' + '|'.join(aliases) + r')\b', line):
                        continue
                    for operand in re.findall(r'\b(?:' + '|'.join(aliases) + r')\s*\(([^)]*)\)', line):
                        match = re.fullmatch(r'\s*(?:"([^"\n]+)"|<([^>\n]+)>)\s*', operand)
                        if match is None:
                            raise ValueError("computed header discovery requires the ordinary compiler")
                        name = match.group(1) or match.group(2)
                        discovery = (name, str(local))
                        if discovery not in discoveries:
                            discoveries.add(discovery)
                            resolve(name, local, bool(match.group(2)))
            if before == (len(texts), len(aliases), len(discoveries)):
                break
        def portable_path(path):
            for i, root in enumerate(roots):
                try:
                    return f"root{i}/" + Path(path).relative_to(root.resolve()).as_posix()
                except ValueError:
                    pass
            return path
        return ([{"path": portable_path(path), "sha256": digest} for path, digest in sorted(found.items())]
                + [{"missing": name} for name in sorted(missing)])
    return [{"path": path, "sha256": digest} for path, digest in sorted(found.items())]
