"""Scan gf.cu for function-scope column arrays.

Reports, per __device__ function, every `float NAME[GF_KP];` / `int NAME[GF_KP];`
declaration at brace depth 1 -- the declarations that ptxas charges to the
per-thread local frame.  Read-only; used to plan and to audit the workspace
conversion.
"""
from __future__ import annotations

import re
import sys

DECL = re.compile(
    r"^\s*(float|int)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*GF_KP\s*\]\s*(;|=)")
MULTI = re.compile(
    r"^\s*(float|int)\s+((?:[A-Za-z_][A-Za-z0-9_]*\s*\[\s*GF_KP\s*\]\s*,\s*)+"
    r"[A-Za-z_][A-Za-z0-9_]*\s*\[\s*GF_KP\s*\])\s*;")


def functions(lines):
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if (line.startswith("__device__") or line.startswith("__global__")
                or line.startswith('extern "C" __global__')):
            m = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", line)
            name = m.group(1) if m else "?"
            j = i
            while j < len(lines) and not lines[j].rstrip().endswith("{"):
                j += 1
                if j - i > 40:
                    break
            if j >= len(lines) or not lines[j].rstrip().endswith("{"):
                i += 1
                continue
            body_start = j
            depth = 1
            k = j
            while depth > 0 and k + 1 < len(lines):
                k += 1
                depth += lines[k].count("{") - lines[k].count("}")
            out.append((name, i, body_start, k))
            i = k
        i += 1
    return out


def main(path):
    lines = open(path, encoding="utf-8").read().split("\n")
    for name, decl_line, body_start, end in functions(lines):
        found = []
        depth = 1
        for n in range(body_start + 1, end + 1):
            line = lines[n]
            if depth == 1:
                m = MULTI.match(line)
                if m:
                    for chunk in m.group(2).split(","):
                        nm = chunk.split("[")[0].strip()
                        found.append((n + 1, m.group(1), nm, "multi"))
                else:
                    m = DECL.match(line)
                    if m:
                        found.append((n + 1, m.group(1), m.group(2),
                                      "init" if m.group(3) == "=" else "plain"))
            depth += line.count("{") - line.count("}")
        if found:
            nf = sum(1 for f in found if f[1] == "float")
            ni = sum(1 for f in found if f[1] == "int")
            print(f"== {name} (lines {decl_line + 1}-{end + 1}): "
                  f"{nf} float + {ni} int GF_KP arrays")
            for ln, ty, nm, kind in found:
                print(f"   {ln:5d} {ty:5s} {nm:20s} {kind}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "gpuwm/core/kernels/gf.cu")
